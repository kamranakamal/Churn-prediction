FROM python:3.11-slim

WORKDIR /app
#Copy only requirements.txt for caching 
COPY requirements.txt ./


#Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy whole directory
COPY . .

#Expose the Port
EXPOSE 8000

#Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

