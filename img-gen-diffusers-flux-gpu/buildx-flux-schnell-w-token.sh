# HF_TOKEN must be an access token that has permission for the model repo
export HF_TOKEN=<YOUR_HF_TOKEN>

docker buildx build \
  --secret id=hf_token,env=HF_TOKEN \   # passes the token without baking it into layers
  --platform linux/amd64 \
  --progress plain \
  -t bytenite/flux-schnell:latest \
  .