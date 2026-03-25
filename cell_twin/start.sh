#!/bin/bash
# Cell Digital Twin — startup script

echo "🧬 Cell Digital Twin v2"
echo "========================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install Python 3.10+"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
cd backend
pip install -r requirements.txt -q

# Start backend in background
echo "🚀 Starting FastAPI backend on http://localhost:8000 ..."
python main.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend
sleep 2

# Serve frontend
echo "🌐 Serving frontend on http://localhost:3000 ..."
cd ../frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!

echo ""
echo "✅ System running:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait and cleanup
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" SIGINT SIGTERM
wait
