Custom wake word detection system trained to recognise the spoken keyword 'Kyro' without the help of common API's. This will be used as a basis for learning how each piece of an LLM works from scratch.
Utilizes libraries such as PyTorch, Librosa, and NumPy.
The model is trained on positive and negative data samples which I have created myself, so far 50 positive and roughly 100 negative, and tested on unseen data that I created to test accuracy.
The project so far is made up of 4 files.
* Record_audio - this script is used to record short audio clips for training and testing but also converts everything to the necessary audio requirements needed for speech recognition i.e. 16-bit PCM, Mono, .wav files.
* Dataset_loader - This loads the files from the respected folders (positive/negative) and performs necessary operations such as MFCC extraction and ensuring the clip is exactly 1 second long before feeding it to our wakeword.
* training_script - this is where the model is trained using the positive/negative/unseen samples that were created and displays a detailed evaluation of how each clip performed.
* Wake_word - this is where the base of our model is. (NEEDS A GOOD DESCRIPTION)  
