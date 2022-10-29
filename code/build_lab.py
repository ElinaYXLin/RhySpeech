import os
from tqdm import tqdm
import yaml



config = yaml.load(open("config/LibriTTS/preprocess.yaml", "r"), Loader=yaml.FullLoader)
in_dir = config["path"]["raw_path"]

speakers = {}
for i, speaker in enumerate(tqdm(os.listdir(in_dir))):
            speakers[speaker] = i

            for chapter in os.listdir(os.path.join(in_dir, speaker)):
                if (chapter == '.DS_Store'):
                    continue

                with open (os.path.join(in_dir, speaker, chapter, f"{speaker}-{chapter}.trans.txt")) as f:
                    content = f.readlines()
                    for ln in content:
                        newLn = ln.split() # newLn is a list, newLn[0] is name, afterwards is content
                        with open(os.path.join(in_dir, speaker, chapter, newLn[0]+".lab"), "w") as g:
                            g.write(ln[len(newLn[0])+1:-1])
