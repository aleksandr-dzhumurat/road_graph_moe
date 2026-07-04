CURRENT_DIR = $(shell pwd)
include .env
export

FULL_IMAGE   = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
NEBIUS       ?= $(HOME)/.nebius/bin/nebius



build-local:
	docker build --platform linux/amd64 -t $(FULL_IMAGE) .

push:
	docker push $(FULL_IMAGE)

build-remote:
	python3 scripts/build_image_remote.py --tag $(IMAGE_TAG) --ssh-key $(SSH_KEY)

create-s3-buckets:
	aws s3 mb s3://geo-trajectories --profile nebius --endpoint-url $(S3_ENDPOINT_URL) || true
	aws s3 mb s3://geo-trajectories-checkpoints --profile nebius --endpoint-url $(S3_ENDPOINT_URL) || true
	aws s3 mb s3://geo-trajectories-road-enhanced --profile nebius --endpoint-url $(S3_ENDPOINT_URL) || true
	aws s3 mb s3://geo-trajectories-code --profile nebius --endpoint-url $(S3_ENDPOINT_URL) || true

create-s3-secret:
	python3 -c "\
import json,subprocess,os,configparser; \
cfg=configparser.ConfigParser(); cfg.read(os.path.expanduser('~/.aws/credentials')); \
k=cfg['nebius']['aws_access_key_id'].strip(); s=cfg['nebius']['aws_secret_access_key'].strip(); \
payload=json.dumps([{'key':'S3_ACCESS_KEY_ID','string_value':k},{'key':'S3_SECRET_ACCESS_KEY','string_value':s}]); \
subprocess.run(['$(NEBIUS)','mysterybox','secret','create','--parent-id','$(PROJECT_ID)','--name','nebius-s3-creds','--secret-version-payload',payload],check=True)"

delete-s3-secret:
	@$(NEBIUS) mysterybox secret list --parent-id $(PROJECT_ID) --format json | \
	  python3 -c "import json,sys,subprocess; [subprocess.run(['$(NEBIUS)','mysterybox','secret','delete','--id',i['metadata']['id']]) for i in json.load(sys.stdin).get('items',[])]"

run-cloud:
	$(NEBIUS) ai job create \
	  --image $(FULL_IMAGE) \
	  --container-command "python -u scripts/backbone.py --batch 64 --steps 300000" \
	  --platform gpu-h100-sxm \
	  --preset 1gpu-16vcpu-200gb \
	  --preemptible \
	  --subnet-id $(SUBNET_ID) \
	  --volume s3://geo-trajectories:/app/data/gold:ro:nebius@nebius-s3-creds \
	  --volume s3://geo-trajectories-checkpoints:/app/data/checkpoints:rw:nebius@nebius-s3-creds \
	  --env S3_ENDPOINT_URL=$(S3_ENDPOINT_URL) \
	  --env PYTHONUNBUFFERED=1

setup:
	mkdir -p data
	@if [ ! -f .env ]; then cp .env.template .env; echo "Created .env from template"; else echo ".env already exists, skipping"; fi
	 UV_VENV_CLEAR=1 uv venv --python 3.13
	uv pip install -r requirements.txt
	@echo ""
	@echo "Done. Activate the venv with: source .venv/bin/activate"
	mkdir -p data/weights data/checkpoints

train-curve:
	PYTHONPATH=$(CURRENT_DIR)/src \
	python scripts/train_curve.py 0.json --out data/train_curves.png --smooth 100

inference:
	PYTHONPATH=$(CURRENT_DIR)/src \
	python scripts/backbone.py --infer

download-ckpt:
	@mkdir -p data/checkpoints
	@if [ -f data/checkpoints/ckpt_final.pt ]; then \
	  echo "ckpt_final.pt already exists, skipping."; \
	else \
	  aws s3 cp s3://geo-trajectories-checkpoints/ckpt_final.pt data/checkpoints/ckpt_final.pt \
	    --profile nebius \
	    --endpoint-url $(S3_ENDPOINT_URL); \
	fi

download-ckpts: download-ckpt
	@mkdir -p data/checkpoints
	@for expert in porto beijing; do \
	  latest=$$(aws s3 ls s3://geo-trajectories-checkpoints/ \
	    --profile nebius --endpoint-url $(S3_ENDPOINT_URL) \
	    | grep "expert_$${expert}_step_" | sort | tail -1 | awk '{print $$4}'); \
	  if [ -z "$$latest" ]; then \
	    echo "No checkpoint found for expert $${expert}, skipping."; \
	  elif [ -f "data/checkpoints/$$latest" ]; then \
	    echo "$$latest already exists, skipping."; \
	  else \
	    echo "Downloading expert $${expert}: $$latest"; \
	    aws s3 cp "s3://geo-trajectories-checkpoints/$$latest" "data/checkpoints/$$latest" \
	      --profile nebius --endpoint-url $(S3_ENDPOINT_URL); \
	  fi; \
	done


ruff:
	ruff check src/ --fix
	ruff format src/

run-jupyter:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	CONFIG_DIR=${CURRENT_DIR}/configs \
	ENV_PATH=${CURRENT_DIR}/.env \
	RUN_ENV=LOCAL \
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8899 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser

train-experts:
	PYTHONPATH=$(CURRENT_DIR)/src \
	python scripts/experts.py --expert porto --batch-size 32 --max-steps 10000

train-experts-osm:
	PYTHONPATH=$(CURRENT_DIR)/src \
	python scripts/experts.py --expert porto --batch-size 32 --max-steps 10000 --use-osm

train-experts-multi:
	PYTHONPATH=$(CURRENT_DIR)/src \
	python scripts/experts.py --multi-tenant --batch-size 32 --max-steps 10000

train-experts-multi-osm:
	PYTHONPATH=$(CURRENT_DIR)/src \
	python scripts/experts.py --multi-tenant --batch-size 32 --max-steps 10000 --use-osm

debug-experts:
	PYTHONPATH=$(CURRENT_DIR)/src \
	python scripts/experts.py --debug --expert porto

debug-experts-osm:
	PYTHONPATH=$(CURRENT_DIR)/src \
	python scripts/experts.py --debug --expert porto --use-osm

upload-code:
	aws s3 sync scripts/ s3://geo-trajectories-code/ \
	  --profile nebius \
	  --endpoint-url $(S3_ENDPOINT_URL)

upload-road-gold:
	aws s3 sync data/road_enhanced_gold/ s3://geo-trajectories-road-enhanced/ \
	  --profile nebius \
	  --endpoint-url $(S3_ENDPOINT_URL)

clear-road-graph:
	aws s3 rm s3://geo-trajectories-road-enhanced/ \
	  --recursive \
	  --profile nebius \
	  --endpoint-url $(S3_ENDPOINT_URL)

run-cloud-phase02:
	$(NEBIUS) ai job create \
	  --image $(FULL_IMAGE) \
	  --container-command "python -u scripts/experts.py --expert porto --use-road-enhanced --batch-size 32 --max-steps 50000" \
	  --platform gpu-h100-sxm \
	  --preset 1gpu-16vcpu-200gb \
	  --preemptible \
	  --subnet-id $(SUBNET_ID) \
	  --volume s3://geo-trajectories-code:/app/scripts:ro:nebius@nebius-s3-creds \
	  --volume s3://geo-trajectories-road-enhanced:/app/data/road_enhanced_gold:ro:nebius@nebius-s3-creds \
	  --volume s3://geo-trajectories-checkpoints:/app/data/checkpoints:rw:nebius@nebius-s3-creds \
	  --env S3_ENDPOINT_URL=$(S3_ENDPOINT_URL) \
	  --env PYTHONUNBUFFERED=1


run-cloud-phase02-multi:
	$(NEBIUS) ai job create \
	  --image $(FULL_IMAGE) \
	  --container-command "torchrun --nproc-per-node=2 scripts/experts.py --multi-tenant --distributed --use-road-enhanced --batch-size 32 --epochs 50" \
	  --platform gpu-l40s-d \
	  --preset 2gpu-64vcpu-384gb \
	  --preemptible \
	  --subnet-id $(SUBNET_ID) \
	  --volume s3://geo-trajectories-code:/app/scripts:ro:nebius@nebius-s3-creds \
	  --volume s3://geo-trajectories-road-enhanced:/app/data/road_enhanced_gold:ro:nebius@nebius-s3-creds \
	  --volume s3://geo-trajectories-checkpoints:/app/data/checkpoints:rw:nebius@nebius-s3-creds \
	  --env S3_ENDPOINT_URL=$(S3_ENDPOINT_URL) \
	  --env PYTHONUNBUFFERED=1

setup-valhalla:
	@echo "Setting up Valhalla with Porto and Beijing data..."
	@echo "Checking prerequisites..."
	@command -v wget >/dev/null 2>&1 || { echo "wget not found, installing via brew..."; brew install wget; }
	@command -v osmium >/dev/null 2>&1 || { echo "osmium-tool not found, installing via brew..."; brew install osmium-tool; }
	mkdir -p data/valhalla_data/input
	@echo "Downloading Portugal OSM data (~180 MB)..."
	wget -O data/valhalla_data/input/portugal.osm.pbf \
	  https://download.geofabrik.de/europe/portugal-latest.osm.pbf
	@echo "Downloading China OSM data (~1.2 GB)..."
	wget -O /tmp/china.osm.pbf \
	  https://download.geofabrik.de/asia/china-latest.osm.pbf
	@echo "Extracting Beijing region from China data..."
	osmium extract \
	  --bbox=116.0,39.6,116.8,40.2 \
	  /tmp/china.osm.pbf \
	  -o data/valhalla_data/input/beijing.osm.pbf
	@echo "Merging Porto and Beijing regions..."
	osmium merge \
	  data/valhalla_data/input/portugal.osm.pbf \
	  data/valhalla_data/input/beijing.osm.pbf \
	  -o data/valhalla_data/input/merged.osm.pbf
	@echo "Creating docker-compose.yml for Valhalla..."
	@echo "services:" > docker-compose.yml
	@echo "  valhalla:" >> docker-compose.yml
	@echo "    image: ghcr.io/valhalla/valhalla:latest" >> docker-compose.yml
	@echo "    ports:" >> docker-compose.yml
	@echo "      - \"8002:8002\"" >> docker-compose.yml
	@echo "    volumes:" >> docker-compose.yml
	@echo "      - ./data/valhalla_data:/data/valhalla" >> docker-compose.yml
	@echo "    command: >" >> docker-compose.yml
	@echo "      bash -c \"" >> docker-compose.yml
	@echo "        echo 'Starting Valhalla setup...' &&" >> docker-compose.yml
	@echo "        ls -la /data/valhalla/input/ &&" >> docker-compose.yml
	@echo "        echo 'Building configuration...' &&" >> docker-compose.yml
	@echo "        valhalla_build_config --mjolnir-tile-dir /data/valhalla/tiles > /data/valhalla/valhalla.json &&" >> docker-compose.yml
	@echo "        echo 'Building tiles from OSM data...' &&" >> docker-compose.yml
	@echo "        valhalla_build_tiles -c /data/valhalla/valhalla.json /data/valhalla/input/merged.osm.pbf &&" >> docker-compose.yml
	@echo "        echo 'Tiles built successfully, starting service...' &&" >> docker-compose.yml
	@echo "        valhalla_service /data/valhalla/valhalla.json 1" >> docker-compose.yml
	@echo "      \"" >> docker-compose.yml
	@echo ""
	@echo "Setup complete! To start Valhalla run: docker compose up"
	@echo "API will be available at http://localhost:8002"
	@echo "Note: First startup will build tiles (~12 min, 4GB RAM required)"
