"""Training-only, two-level scalar codecs for one 384-dimensional embedding.

This is a separate supplementary experiment. It does not change the frozen
primary retrieval experiment. There are no labels, IDs, queries or relevance
judgments in this module's fit API.
"""
from dataclasses import dataclass
import struct
import numpy as np

WIDTH = 384
HEADER = struct.Struct('<2sBBHH')
KINDS = {'centered_symmetric': 1, 'raw_two_means': 2,
         'pca_symmetric': 3, 'pca_two_means': 4}
NAMES = {v: k for k, v in KINDS.items()}
PARAMETER_FLOATS = {1: 768, 2: 768, 3: 148224, 4: 148608}


@dataclass(frozen=True)
class Asset:
    name: str
    mean: object = None
    basis: object = None
    amplitude: object = None
    low: object = None
    high: object = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_array(value, shape):
    value = np.array(value, dtype='<f4', copy=True, order='C')
    require(value.shape == shape and np.isfinite(value).all(), 'Invalid parameter array')
    value.setflags(write=False)
    return value


def validate(asset):
    require(isinstance(asset, Asset) and asset.name in KINDS, 'Unknown asset')
    kind = KINDS[asset.name]
    fields = set()
    if kind in (1, 3, 4):
        fields.add('mean')
    if kind in (3, 4):
        fields.add('basis')
    fields |= {'amplitude'} if kind in (1, 3) else {'low', 'high'}
    for key in ('mean', 'basis', 'amplitude', 'low', 'high'):
        value = getattr(asset, key)
        if key not in fields:
            require(value is None, 'Unexpected asset parameter')
            continue
        shape = (WIDTH, WIDTH) if key == 'basis' else (WIDTH,)
        require(isinstance(value, np.ndarray) and value.shape == shape and
                value.dtype == np.dtype('<f4') and np.isfinite(value).all(), 'Invalid ' + key)
    if kind in (1, 3):
        require(np.all(asset.amplitude >= 0), 'Negative amplitude')
    else:
        require(np.all(asset.low <= asset.high), 'Reversed two-means levels')
    return kind


def two_means(values):
    """Global minimum-SSE split per coordinate; ties use first sorted split.

    Sort each 1-D training coordinate, evaluate every split between distinct
    adjacent values using centered prefix sums, then store the cluster means
    as float32. This has no initialization, iterative tuning or labels.
    """
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] == WIDTH
            and np.isfinite(values).all(), 'Invalid calibration values')
    low, high = np.empty(WIDTH), np.empty(WIDTH)
    n = len(values)
    left_n = np.arange(1, n, dtype=np.float64)
    right_n = n - left_n
    for j in range(WIDTH):
        origin = values[:, j].mean()
        original_ordered = np.sort(values[:, j], kind='stable')
        ordered = original_ordered - origin
        if ordered[0] == ordered[-1]:
            low[j] = high[j] = values[0, j]
            continue
        total = np.cumsum(ordered, dtype=np.float64)
        squared = np.cumsum(ordered * ordered, dtype=np.float64)
        losses = squared[:-1] - total[:-1] ** 2 / left_n
        losses += (squared[-1] - squared[:-1]) - (total[-1] - total[:-1]) ** 2 / right_n
        losses[ordered[:-1] == ordered[1:]] = np.inf
        cut = int(np.argmin(losses)) + 1
        low[j] = original_ordered[:cut].mean(dtype=np.float64)
        high[j] = original_ordered[cut:].mean(dtype=np.float64)
    return frozen_array(low, (WIDTH,)), frozen_array(high, (WIDTH,))


def fit(calibration, pca_mean, pca_basis):
    x = np.asarray(calibration)
    require(x.dtype == np.dtype('<f4') and x.ndim == 2 and x.shape[1] == WIDTH
            and x.shape[0] >= 2 and np.isfinite(x).all(), 'Invalid calibration matrix')
    # Reuse the already frozen full-rank training PCA. The runner binds it to
    # the same 5,055 training rows and the original preparation receipt.
    mean = frozen_array(pca_mean, (WIDTH,))
    basis = frozen_array(pca_basis, (WIDTH, WIDTH))
    require(np.array_equal(x.mean(axis=0).astype('<f4'), mean), 'PCA mean does not match calibration')
    require(np.allclose(basis.astype(np.float64) @ basis.astype(np.float64).T,
                        np.eye(WIDTH), rtol=0, atol=2e-5), 'Basis is not orthonormal')
    centered = x.astype(np.float64) - mean.astype(np.float64)
    amplitude = frozen_array(np.abs(centered).mean(axis=0), (WIDTH,))
    low, high = two_means(x)
    # Encoder uses the stored float32 transform, so fit its actual coordinates.
    coordinates = (x - mean) @ basis.T
    pca_amplitude = frozen_array(np.abs(coordinates.astype(np.float64)).mean(axis=0), (WIDTH,))
    pca_low, pca_high = two_means(coordinates)
    result = {
        'centered_symmetric': Asset('centered_symmetric', mean=mean, amplitude=amplitude),
        'raw_two_means': Asset('raw_two_means', low=low, high=high),
        'pca_symmetric': Asset('pca_symmetric', mean=mean, basis=basis, amplitude=pca_amplitude),
        'pca_two_means': Asset('pca_two_means', mean=mean, basis=basis, low=pca_low, high=pca_high),
    }
    for asset in result.values():
        validate(asset)
    return result


def dump_asset(asset):
    kind = validate(asset)
    arrays = []
    if kind in (1, 3, 4):
        arrays.append(asset.mean)
    if kind in (3, 4):
        arrays.append(asset.basis)
    arrays.extend([asset.amplitude] if kind in (1, 3) else [asset.low, asset.high])
    payload = b''.join(a.astype('<f4', copy=False).tobytes(order='C') for a in arrays)
    require(len(payload) == PARAMETER_FLOATS[kind] * 4, 'Asset serialization size')
    return HEADER.pack(b'QA', 1, kind, WIDTH, 0) + payload


def load_asset(data):
    require(isinstance(data, bytes) and len(data) >= 8, 'Short asset')
    magic, version, kind, width, reserved = HEADER.unpack(data[:8])
    require(magic == b'QA' and version == 1 and kind in NAMES and width == WIDTH
            and reserved == 0, 'Invalid asset header')
    require(len(data) == 8 + PARAMETER_FLOATS[kind] * 4, 'Invalid asset length')
    values = np.frombuffer(data[8:], dtype='<f4')
    require(np.isfinite(values).all(), 'Nonfinite asset parameter')
    offset = 0
    def take(shape):
        nonlocal offset
        count = int(np.prod(shape))
        result = frozen_array(values[offset:offset + count].reshape(shape), shape)
        offset += count
        return result
    parameters = {}
    if kind in (1, 3, 4):
        parameters['mean'] = take((WIDTH,))
    if kind in (3, 4):
        parameters['basis'] = take((WIDTH, WIDTH))
    if kind in (1, 3):
        parameters['amplitude'] = take((WIDTH,))
    else:
        parameters['low'], parameters['high'] = take((WIDTH,)), take((WIDTH,))
    asset = Asset(NAMES[kind], **parameters)
    validate(asset)
    if kind in (3, 4):
        require(np.allclose(asset.basis.astype(np.float64) @ asset.basis.astype(np.float64).T,
                            np.eye(WIDTH), rtol=0, atol=2e-5), 'Basis is not orthonormal')
    return asset


def encode(x, asset):
    kind = validate(asset)
    x = np.asarray(x, dtype='<f4')
    require(x.shape == (WIDTH,) and np.isfinite(x).all(), 'Invalid query vector')
    if kind in (3, 4):
        values = (x - asset.mean) @ asset.basis.T
    elif kind == 1:
        values = x - asset.mean
    else:
        values = x
    if kind in (1, 3):
        bits = values >= 0
    else:
        midpoint = (asset.low.astype(np.float64) + asset.high.astype(np.float64)) / 2
        bits = values.astype(np.float64) >= midpoint
    require(np.isfinite(values).all(), 'Nonfinite projected query')
    return HEADER.pack(b'BQ', 1, kind, WIDTH, 0) + np.packbits(bits, bitorder='little').tobytes()


def decode(packet, asset):
    kind = validate(asset)
    require(isinstance(packet, bytes) and len(packet) == 56, 'Packet must be exactly56bytes')
    magic, version, packet_kind, width, reserved = HEADER.unpack(packet[:8])
    require((magic, version, packet_kind, width, reserved) == (b'BQ', 1, kind, WIDTH, 0), 'Packet/asset mismatch')
    bits = np.unpackbits(np.frombuffer(packet[8:], dtype=np.uint8), bitorder='little').astype(bool)
    if kind in (1, 3):
        values = np.where(bits, asset.amplitude, -asset.amplitude)
    else:
        values = np.where(bits, asset.high, asset.low)
    if kind in (3, 4):
        result = values @ asset.basis + asset.mean
    elif kind == 1:
        result = values + asset.mean
    else:
        result = values
    require(result.shape == (WIDTH,) and np.isfinite(result).all(), 'Nonfinite reconstructed query')
    return np.asarray(result, dtype='<f4')
