# CS231n Spring 2017 homework
My implementation about [CS231n spring 2017 homework](http://cs231n.github.io/).
I chose to use tensorfow to implement assignment3. Some weird things happend in DCGAN and styletransfer.

1. In questions about DCGAN, all the images in a single step are the same, although all the checks seem to be right.
2. In questions about styletransfer, my code can't synthesis images similar to results in the slides. It seems that my network dosen't learn to the right style.

Hoping someone can pull me out of the mire.

DCGAN error result

![DCGAN error result](https://raw.githubusercontent.com/Psunshine/CS231n-Spring-2017-Assignment/master/.dcgan_err.png) 

DCGAN right result

![DCGAN right result](https://raw.githubusercontent.com/Psunshine/CS231n-Spring-2017-Assignment/master/.dcgan_right.png) 

styletransfer error result

![style error result](https://raw.githubusercontent.com/Psunshine/CS231n-Spring-2017-Assignment/master/.style_err.png) 

styletransfer right result

![style right result](https://raw.githubusercontent.com/Psunshine/CS231n-Spring-2017-Assignment/master/.style_right.png)

--------------
自己完成的[CS231n spring 2017 homework](http://cs231n.github.io/)。
assignment3使用tensorflow实现，但是有两处结果不太对：

1. DCGAN一问中每一步结果都是一样的。而奇怪的是前面的check都通过了。
2. styletransfer一问中得到的图像和课件上的不一样，应该是没有正确转换style。



试验了几天也没找到原因，希望有明白的大佬可以提点提点
