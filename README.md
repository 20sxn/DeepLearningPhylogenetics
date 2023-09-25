# DeepLearningPhylogenetics

## Presentation
This project was done during my internship at the Museum National d'Histoire Naturelle. This project had 3 goals:
1. Improve the methods of estimating the probability of substitution in sequences of amino acids
2. Analyze the importance of each part of the input data
3. Use our trained models for pairwise sequence alignment

## Project Structure
```
├── LICENSE
├── README.md
├── report
│   └── Report.pdf                          (report in french)
└── src
    ├── pairwise_alignment                  (contains all the files needed for pairwise alignement)
    │   ├── align.py                        (use this file to align pairs of sequences)
    │   ├── dataloading_utils_CrossCorr.py  (utils for dataloading during training)
    │   ├── models_CrossCorr_utils.py       (model architecture)
    │   ├── out                             (folder for the output of **align.py**)
    │   └── train_CrossCorr.py              (trains a new model)
    └── substitution_estimation             (contains models for substituton probability estimation)
        ├── baselines                       (baselines such as the LG matrix)
        ├── basemodel                       (our base models)
        ├── data                            (contains the data and some other useful files)
        ├── models                          (trained models will be there)
        ├── NNdist_Qmat                     (model using neural networks and rate matrix)
        ├── NNpid                           (model using neural networks and conditional frequencies)
        └── profile                         (our base model using the profile information)
```

## Data
We used the data from Pfam 35.0 to train our models. If you wish to use the same data as us, download the alignements from Pfam. There is a list of the alignement used in training, testing and validation. Feel free to use other alignements, but you will need to delete all the **dataset.pth** files.

## Substitution probability estimation
To train a specific model, use the respective **train_XXX.py** file.
To use a model, load it using 
```py
savepath = Path(fname)
    if savepath.is_file():
        with savepath.open("rb") as fp:
            state = torch.load(fp)
``` 
And to estimate the substitution probability, use:
```py
y_hat = torch.nn.functional.softmax(state.model(X,...),dim=1)
```
With ```X``` being the batched and required input of the model. 


## Pairwise sequence alignment
To align a set of pairs of sequences, use 
```console 
python align.py in out
```
With ```in``` being the name of the folder containing each files corresponding to a pair of unaligned sequences. And ```out``` being the name of the output folder.

Any input file should only contain 2 unaligned sequences using the FASTA format, for example:
```
>1a0cA
NKYFENVSKIKYEGPKSNNPYSFKFYNPEEVIDGKTMEEHLRFSIAYWHTFTADGTDQFGKATMQRPWNHYTDPMDIAKA
RVEAAFEFFDKINAPYFCFHDRDIAPEGDTLRETNKNLDTIVAMIKDYLKTSKTKVLWGTANLFSNPRFVHGASTSCNAD
VFAYSAAQVKKALEITKELGGENYVFWGGREGYETLLNTDMEFELDNFARFLHMAVDYAKEIGFEGQFLIEPKPKEPTKH
QYDFDVANVLAFLRKYDLDKYFKVNIEANHATLAFHDFQHELRYARINGVLGSIDANTGDMLLGWDTDQFPTDIRMTTLA
MYEVIKMGGFDKGGLNFDAKVRRASFEPEDLFLGHIAGMDAFAKGFKVAYKLVKDRVFDKFIEERYASYKDGIGADIVSG
KADFRSLEKYALERSQIVNKSGRQELLESILNQYLFA
>1a0dA
PYFDNISTIAYEGPASKNPLAFKFYNPEEKVGDKTMEEHLRFSVAYWHTFTGDGSDPFGAGNMIRPWNKYSGMDLAKARV
EAAFEFFEKLNIPFFCFHDVDIAPEGETLKETYKNLDIIVDMIEEYMKTSKTKLLWNTANLFTHPRFVHGAATSCNADVF
AYAAAKVKKGLEIAKRLGAENYVFWGGREGYETLLNTDMKLELDNLARFLHMAVDYAKEIGFDGQFLIEPKPKEPTKHQY
DFDVATALAFLQTYGLKDYFKFNIEANHATLAGHTFEHELRVARIHGMLGSVDANQGDMLLGWDTDEFPTDLYSTTLAMY
EILKNGGLGRGGLNFDAKVRRGSFEPEDLFYAHIAGMDSFAVGLKVAHRLIEDRVFDEFIEERYKSYTEGIGREIVEGTA
DFHKLEAHALQLGEIQNQSGRQERLKTLLNQYLLEVC
```

## Depedencies

This project was built using the following libraries:
```
torch
einops
numpy
linecache
```
