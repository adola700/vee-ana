apt install python3-pip -y
apt install python3.12-venv -y

python3 -m venv venv
source venv/bin/activate

sudo apt update -y
sudo apt install portaudio19-dev python3-dev -y
pip install pyaudio

pip install -r requirements.txt