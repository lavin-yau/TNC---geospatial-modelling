FROM condaforge/miniforge3:24.11.3-0

WORKDIR /app

COPY binder/environment.yml /tmp/environment.yml

RUN mamba env update -n base -f /tmp/environment.yml \
    && mamba clean --all --yes

COPY scripts/ ./scripts/
COPY ML_Modeling_Files/ ./ML_Modeling_Files/

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/matplotlib local_outputs

CMD ["python", "scripts/User.py"]
