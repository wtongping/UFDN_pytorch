# UFDN_pytorch
#### Origin implement link is [here](https://github.com/Alexander-H-Liu/UFDN)</br>
#### This repository modified dataloader, u can load data from image floder rather than h5files.</br>
#### The data should be organized like this:</br>
data</br>
&nbsp; &nbsp; |---dataset_name</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|---train</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;|---domain_A_name</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;1.jpg</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;|---domain_B_name</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;1.jpg</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;|---domain_C_name</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;1.jpg</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|---test</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;|---domain_A_name</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;1.jpg</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;|---domain_B_name</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;1.jpg</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;|---domain_C_name</br>
&nbsp; &nbsp; |&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;1.jpg</br>
