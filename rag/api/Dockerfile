FROM amazon/aws-lambda-python:3.11
WORKDIR /var/task/lambda-chat

COPY __init__.py /var/task/
COPY app.py /var/task/  
COPY requirements.txt /var/task/
COPY . ..


RUN /var/lang/bin/python3.11 -m pip install --upgrade pip
RUN /var/lang/bin/python3 -m pip install -r /var/task/requirements.txt -U --no-cache-dir


RUN pip install -U /var/task/bedrock-sdk/boto3-1.28.21-py3-none-any.whl
RUN pip install -U /var/task/bedrock-sdk/botocore-1.31.21-py3-none-any.whl 
RUN rm -rf /var/task/bedrock-sdk


CMD ["app.handler"]

