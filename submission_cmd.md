bash scp_act.sh
python patch_metadata.py

pixi install
pixi update

docker compose -f docker/docker-compose.yaml build --no-cache model
docker compose -f docker/docker-compose.yaml up

docker tag aic-act-submission:080000 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/extrinsic-ai:v2
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/extrinsic-ai:v2