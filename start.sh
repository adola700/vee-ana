apt install python3-pip
apt install python3.12-venv

python3 -m venv venv
source venv/bin/activate

sudo apt update
sudo apt install portaudio19-dev python3-dev
pip install pyaudio

pip install -r requirements.txt