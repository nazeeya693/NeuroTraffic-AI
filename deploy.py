from pyngrok import ngrok
import os

# Start Streamlit app
os.system("streamlit run app.py &")

# Create public URL
public_url = ngrok.connect(8501)

print("Your Public URL:")
print(public_url)