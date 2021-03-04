# User feedback Results
This projects shows the results of the supervised and unsupervised analysis made using HuliPractice
comments. This project is deployed to AWS ECR using the next code.
```
# All will be pushed as 3.0.0
alias aws='docker run --rm -it -v ~/PycharmProjects/user-feedback/.aws:/root/.aws -e AWS_CONFIG_FILE=/root/.aws amazon/aws-cli'
aws ecr list-images --repository-name=huli-user-feedback --region=us-east-1
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 813076059213.dkr.ecr.us-east-1.amazonaws.com
docker push 813076059213.dkr.ecr.us-east-1.amazonaws.com/huli-user-feedback:3.0.0
```

## Acknowledgments
1. Users comments are not uploaded to the project for security reasons.
2. Main application code can be found at app.py
3. .aws file is no uploaded to the project for security reasons.