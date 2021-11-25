# syntax=docker/dockerfile:1

FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY pipeline_rdf2vec_mlp.py pipeline_rdf2vec_mlp.py
CMD ["mkdir", "test_files"]
COPY test_files/res1_entities_SMALL.tsv test_files/res1_entities_SMALL.tsv
COPY test_files/res1_hp_temp_kg_SMALL.ttl test_files/res1_hp_temp_kg_SMALL.ttl

run pip3 install -r requirements.txt

CMD [ "python3", "pipeline_rdf2vec_mlp.py"]