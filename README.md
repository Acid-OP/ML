1. Git clone the project 
2. if uv is not install do pip install uv in your machine (if pip also not download gpt the command to download it) after that do uv sync in the terminal where you git cloned it 
3. add cuda in your machine( go to google and type download cuda 12.6 version downlaod it)
4. install it in your machine 
5. come to vs code where you git cloned the project 
6. run this command in terminal - source .venv/Scripts/activate
7. In the terminal write this command after that- uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
7. let everything get installed 
8. now first run this snippet in any of the file do ctrl v tehre of this snippet just for testing 
import torch

print("CUDA available? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected by PyTorch") 

The output should be that gpu detected like your code should run on your gpu not cpu thats the goal
9. once its verified that its on gpu . run the fine tune file (command maybe uv run python finetune.py) or gpt it 
10. please fix the paths of the files loaded in the code  ( they are written according to my pc setup)
11. then run the second test file 
12. right now there is one image only change the path with any image (of the dataset or of yours anything just write the right path) to test it is working fine it is telling non for non mangroove and mangroove for mangroove .
13. you can test it on feeding the whole folder also for testing not just a singler image (i did that because our app will work like that) just gpt it for that



TODO- 1. BETTER DATASET (anyone find some better dataset for the rest of the crops i.e seagrass, saltmarsh or this mangroove also for better performance)
2. Research work of what these crops needed how much land and all those things that we can add in the modal to check weather the uploaded image/coordinate is true or not.
3. And test it and tell how this current is working (it is fine tuned on very less data) but working fine acc to me . Rest check it out and tell me (for now this is just for mmangroove)