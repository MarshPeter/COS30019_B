# COS30019_B

## Steps to work locally

1. Create a local environment so you don't install everything globally.

Do this by using the following commands on powershell in order:

```powershell
py -m venv myenv
myenv\Scripts\Activate.ps1
```

This will put you into your virtual environment, and all stuff downloaded will be only included to this environment. Without this all libraries would be installed globally onto your computer and you may eventually run into problems with installations attempting to over name each other.

2. Install the required libraries located within requirements.txt

```powershell
pip install -r requirements.txt
```

Make sure the installation happens within your virtual environment. Once installed, they will be limited to your local environment

## Additional tips

If you wish to leave the virtual enviroment, simply type "decativate" in the console.
If you aren't sure if you downloaded everything, you can see everything downloaded using "pip list"
