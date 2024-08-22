# evaluate-QandA-from-document-streamlit


## Getting Started
**Prerequisites**
Before running the application, ensure you have the following installed:
- Python 3.7 or higher
- pip for package management

### Installation
- Clone the repository:
```
git clone https://github.com/your-username/streamlit-redaction-app.git
cd streamlit-redaction-app
```

- Install required packages:
```
pip install -r requirements.txt
```

- Set up OpenAI API key:
    - Obtain your API key from OpenAI.
    - Set up an environment variable OPENAI_API_KEY:
    ```
    export OPENAI_API_KEY='your-api-key-here'
    ```
- Running the Application
To start the application, navigate to the project directory and run:
```
streamlit run main.py 
```

This command will launch the application in your default web browser.

### Deploying the Application
Streamlit Cloud is a simple and free way to deploy your Streamlit applications. Follow these steps to deploy:

- Sign up or log in to Streamlit Cloud:
   - Go to [Streamlit Cloud](https://streamlit.io/) and sign up or log in.

- Create a new app:
    - Click on "New app" and connect your GitHub repository containing the Streamlit application.

- Configure your deployment:
    - Repository: Select your GitHub repository.
    - Branch: Choose the branch you want to deploy (e.g., main).
    - File path: Enter the path to your Streamlit app file (e.g., app.py).

- Set environment variables:
    - In the "Advanced settings" of your app, add your OpenAI API key as an environment variable:
    ```
    Key: OPENAI_API_KEY
    Value: your-api-key-here
    ```
- Deploy the app:
    - Click "Deploy" and Streamlit will handle the rest. Once the deployment is complete, your app will be live and accessible via a unique URL.
