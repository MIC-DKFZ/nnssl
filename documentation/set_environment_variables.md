# How to set environment variables

nnssl requires you to set some environment variables so that it always knows where raw data, preprocessed data and trained 
models are. Depending on the operating system, these environment variables need to be set in different ways.

Variables can either be set permanently (recommended!) or you can decide to set them every time you call nnssl. 

# Linux & MacOS

## Permanent
Locate the `.bashrc` file in your home folder and add the following lines to the bottom:

```bash
export nnssl_raw="/Your/Path/To/nnssl_raw"
export nnssl_preprocessed="/Your/Path/To/nnssl_preprocessed"
export nnssl_results="/Your/Path/To/nnssl_results"
```

(Of course you need to adapt the paths to the actual folders you intend to use).
If you are using a different shell, such as zsh, you will need to find the correct script for it. For zsh this is `.zshrc`.

## Temporary
Just execute the following lines whenever you run nnU-Net:
```bash
export nnssl_raw="/Your/Path/To/nnssl_raw"
export nnssl_preprocessed="/Your/Path/To/nnssl_preprocessed"
export nnssl_results="/Your/Path/To/nnssl_results"
```
(Of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your terminal! They will also only apply to the current 
terminal window and DO NOT transfer to other terminals!

Alternatively you can also just prefix them to your nnU-Net commands:

`nnssl_results="/Your/Path/To/nnssl_results" nnssl_preprocessed="/Your/Path/To/nnssl_preprocessed" nnUNetv2_train[...]`

## Verify that environment parameters are set
You can always execute `echo ${nnssl_raw}` etc to print the environment variables. This will return an empty string if 
they were not set correctly.

# Windows
Useful links:
- [https://www3.ntu.edu.sg](https://www3.ntu.edu.sg/home/ehchua/programming/howto/Environment_Variables.html#:~:text=To%20set%20(or%20change)%20a,it%20to%20an%20empty%20string.)
- [https://phoenixnap.com](https://phoenixnap.com/kb/windows-set-environment-variable)

## Permanent
See `Set Environment Variable in Windows via GUI` [here](https://phoenixnap.com/kb/windows-set-environment-variable). 
Or read about setx (command prompt).

## Temporary
Just execute the following before you run nnssl:

(PowerShell)
```PowerShell
$Env:nnssl_raw = "C:/Users/YourName/nnssl_raw"
$Env:nnssl_preprocessed = "C:/Users/YourName/nnssl_preprocessed"
$Env:nnssl_results = "C:/Users/YourName/nnssl_results"
```

(Command Prompt)
```Command Prompt
set nnssl_raw=C:/Users/YourName/nnssl_raw
set nnssl_preprocessed=C:/Users/YourName/nnssl_preprocessed
set nnssl_results=C:/Users/YourName/nnssl_results
```

(Of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your session! They will also only apply to the current 
window and DO NOT transfer to other sessions!

## Verify that environment parameters are set
Printing in Windows works differently depending on the environment you are in:

PowerShell: `echo $Env:[variable_name]`

Command Prompt: `echo %[variable_name]%`
