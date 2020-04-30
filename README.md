![logo](images/QuTiBench_Logo.png)

# QuTiBench
Neural Networks have become one of the most successful universal machine learning algorithms. 
They play a key role in enabling machine vision and speech recognition, and are increasingly 
adopted in other application domains. 
While the underlying computation is structurally simple, their computational complexity is enormous 
and comes along with equally challenging memory requirements both in regards to capacity and access bandwidth. 
This limits deployment in particular within energy constrained, embedded environments.  
In order to address these implementation challenges, a broad spectrum of new customized and heterogeneous 
hardware architectures have emerged, sometime referred to as deep learning processing units, 
often accompanied with co-designed algorithms to extract maximum benefit out of the hardware. 
Furthermore, numerous optimization techniques are being explored to reduce 
compute and memory requirements while maintaining accuracy.
This results in an abundance of algorithmic and architectural choices, some of which fit specific use cases 
better than others.  

For system level designers, there is currently no good way to compare the variety of hardware, algorithm and 
optimization options. While there are many benchmarking efforts in this field, they cover only subsections of 
the embedded design space.  None of the existing benchmarks support essential algorithmic optimizations such as 
quantization, an important technique to stay on chip, or specialized heterogeneous hardware architectures. 
We propose a novel benchmark suite, named QuTiBench, that addresses this need.  
QuTiBench is a novel multi-tiered benchmarking methodology that supports algorithmic optimizations such as 
quantization and helps system developers understand the benefits and limitations of these novel compute architectures 
in regards to specific neural networks and will help drive future innovation.  
We invite the community to contribute to QuTiBench in order to be able to support the full spectrum of choices 
in implementing machine learning systems.

# Contributing
See the [website](https://rcl-lab.github.io/qutibench/contributing/meta/2020/04/09/Contributing.html) for instructions on contributing.


# About us
![Michaela Blott](images/michaela_blott.png)
Michaela Blott is the principal engineer at Xilinx Inc. She has more than 20 years of experience in FPGA and board-level design and networking. Her main areas of interest include data centers, high-level synthesis tools, and high-speed network processing on FPGAs. Asked about the biases that women in technology still confront, she says she faces less pressure to justify her existence in tech than to justify her decision to be a working mother.
 
![Miriam Leeser](images/miriam_leeser.png) 
Miriam Leeser is a Professor in Electrical and Computer Engineering at Northeastern where she is head of the Reconfigurable and GPU Computing Laboratory.  She received her BS degree in electrical engineering from Cornell University, and Diploma and PhD degrees in computer science from Cambridge University in England. She was on the faculty at Cornell University’s Department of Electrical Engineering before coming to Northeastern, where she is a member of the Computer Engineering research group. Her research includes using heterogeneous architectures for signal and image processing applications including wireless communications as well as implementing computer arithmetic and verifying critical applications. She is a senior member of the IEEE and of the ACM.
    
![Zachary Neveau](images/zachary_neveau.png)
Zachary Neveau is a student.

![Alina Vasilciuc](images/alina_vasilciuc.png)
Alina Vasilciuc is a student.

# Publications
Blott, Michaela, et al. "QuTiBench: Benchmarking neural networks on heterogeneous hardware." 
ACM Journal on Emerging Technologies in Computing Systems (JETC) 15.4 (2019): 1-38.
https://arxiv.org/pdf/1909.05009.pdf

Michaela Blott, Johannes Kath, Lisa Halder, Yaman Umuroglu, Nicholas Fraser, Giulio Gambardella, 
Miriam Leeser, and Linda Doyle. 2020. 
"Evaluation of Optimized CNNs on FPGA and non-FPGA based Accelerators using a Novel Benchmarking Approach." In The 
2020 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA ’20).
Association for Computing Machinery, New York, NY, USA, 317. 
https://dl.acm.org/doi/10.1145/3373087.3375348
Poster available here: https://github.com/michaelablott/QuTiBench/blob/master/Publications/FPGA2020_EvalCNNs_Poster.pdf

## Webmaster Info
All info on how to customize the site is located on the [fastpages github](https://github.com/fastai/fastpages). The platform is fairly new, so as changes are added to the faspages repo it's not a bad idea to update this website to reflect that. Updating instructions are available also through the fastpages github.