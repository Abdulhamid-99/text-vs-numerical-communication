FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

WORKDIR /code
ENV PYTHONPATH=/code/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PIP_DEFAULT_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-model.txt /requirements-model.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install pip==25.3 \
 && python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 \
 && python -m pip install -r /requirements-model.txt \
 && python -m pip check
COPY runtime/ /code/
RUN python /code/thesis_study/study_worker.py --help
ENTRYPOINT ["python"]
CMD ["/code/thesis_study/study_worker.py", "--help"]
