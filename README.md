# Face_Recognition_using_SQLite_Database

At first make shore that you installed the required modules, use commands provided in requirements.txt file.
Then create two directory/folder "dataset" and "recognizer".
dataset_creator.py file holds the code for adding the new user and collecting the photo samples of the user.
Then replace the path of xml and db file with relative path of respective files.
This collects the details of user like id, name, age & photo samples.

The second step is to train the photo samples.
trainer.py file holds the code to train the photo samples present in dataset directory.
Then replace the path of dataset to relative path of dataset created after first step.
The YML model is created in recognizer directory after the execution of code.

The third step is to detect the face.
detect.py file holds the code to detect the faces of the user.
Replace the path of xml, yml & db file to the relative path of respective file.
Use q to quit the process.
