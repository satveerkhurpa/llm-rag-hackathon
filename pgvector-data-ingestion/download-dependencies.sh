#!/bin/sh

set -e

#sudo yum install -y unzip

#Download the Bedrock-specific libraries
curl https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.zip --output bedrock-python-sdk.zip
unzip bedrock-python-sdk.zip -d bedrock-python-sdk


