# user-feedback
User feedback deployment proyect

Commands to push image to ECR
```
# All will be pushed as latest
alias aws='docker run --rm -it -v ~/PycharmProjects/user-feedback/.aws:/root/.aws -e AWS_CONFIG_FILE=/root/.aws amazon/aws-cli'
aws ecr list-images --repository-name=huli-user-feedback --region=us-east-1
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 813076059213.dkr.ecr.us-east-1.amazonaws.com
docker push 813076059213.dkr.ecr.us-east-1.amazonaws.com/huli-user-feedback:2.0.0
```