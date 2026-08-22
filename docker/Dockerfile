# The whole repository in a container, not just the code.
#
# Carrying only src/ is the tempting version and it breaks: the tests check the
# README against the committed results, the config files drive every command, and
# the analysis output has to land somewhere a volume can pick it up. So the image
# holds the repository.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=2

WORKDIR /app

# Dependencies first, so a source edit does not reinstall scipy.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && pip install -r requirements-dev.txt

COPY pyproject.toml README.md LICENSE Makefile ./
COPY src ./src
COPY tests ./tests
COPY config ./config
COPY scripts ./scripts
COPY docs ./docs
COPY analysis ./analysis
COPY results ./results
COPY artifacts ./artifacts

RUN pip install --no-deps -e .

# A non-root user, because the container writes into /app and a root-owned
# analysis/ directory on a bind mount is a nuisance to clean up afterwards.
RUN useradd --create-home --uid 1000 cobalt && chown -R cobalt:cobalt /app
USER cobalt

# Prove the image is usable rather than merely built. This runs at build time,
# so a container that cannot import its own package fails here rather than in
# whatever consumes it.
RUN python -c "import cobalt; print('cobalt', cobalt.__version__)"

EXPOSE 8000

# The default is the demo: `docker compose up` reproduces every number.
CMD ["bash", "scripts/run_demo.sh"]
