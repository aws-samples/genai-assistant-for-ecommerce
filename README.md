# Build E-commerce Assistant with Genai

This repo demonstrates how to use Amazon Bedrock and generative AI to boost business innovation and efficiency through natural language interactions.

This sample code showcases a Streamlit frontend where users can input requests for various use cases, including text-to-image, image-to-image, and image question-answering.

# How to deploy:

## Prerequisites:
1. Amazon Bedrock Access.
2. Ensure Python 3.9 or above version installed. it can be downloaded [here](https://www.python.org/downloads/release/python-3911/).

Step 1: Clone this repository

Step 2: Set up and activate a Python 3.9 virtual environment

```
pip install virtualenv
python3.9 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Step 3: Install requirements

```
pip install -r requirements.txt
```
Step 4: Configure environment variables

Create a .env file in the root directory with the following content:

```
save_folder=<PATH_TO_ROOT_OF_THIS_REPO>
```
Step 5: Run the application

```
streamlit run Home.py
```
You can now upload images or ask text questions to generate responses.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

