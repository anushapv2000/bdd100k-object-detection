FROM python:3.9


# Create working directory
WORKDIR /app

# Copy only requirements first (enables Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for Dash app
EXPOSE 8050

ENV PYTHONUNBUFFERED=1

# Start your Dash application
CMD ["python", "dashboard.py"]
