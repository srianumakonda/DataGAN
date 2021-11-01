# DataGAN: A way to train self-driving tech using synthetic data.

Article: <a href="https://srianumakonda.medium.com/datagan-leveraging-synthetic-data-for-self-driving-vehicles-6e629968a567">https://srianumakonda.medium.com/datagan-leveraging-synthetic-data-for-self-driving-vehicles-6e629968a567</a>

So far, more than <a href="https://www.caranddriver.com/news/a30857661/autonomous-car-self-driving-research-expensive/">$16 billion</a> has been spent on self-driving research. What’s the problem? Self-driving is expensive? Why? Getting data and training these models are not only time-consuming but really expensive. You then also need to take into account the fact that Waymo’s spent more than 20 million miles on public roads for data gathering (talking about the amount of energy consumed is for a whole other article…)
What percentage of data is actually useful? Very minimal. Why? Most of the data is usually from “normal” driving scenes, not edge case scenarios such as car overtaking, parking, traffic, etc.

How do we solve that?

General adversarial networks. Focusing on road/scenes rather than edge case scenarios i.e. parking and overtaking. Being able to generate vasts amounts of self-driving data while not having to spend excessive amounts of capital gathering data can be crucial in self-driving deployment.
As an attempt to solve this problem, I’ve been focusing on building DataGAN out. Leveraging Generative Adversarial Networks to create self-driving data at scale is crucial. By emphasizing a focus on DCGANs, I’m focusing on creating high-quality self-driving images that can be used to train and improve the performance of computer vision models such as lane + object detection, and semantic segmentation.

Code is being ran on my laptop (Dell XPS 15 9650, 16GB RAM, NVIDIA GTX 1650 Ti w/ 4GPU RAM), really hard to train haha. Currently training on a smaller version of Cityscapes, image size is resized to 128x128, batch size set as 64, lr is 2e-4 for generator and 1e-5 for discriminator (following principles of <a href="https://arxiv.org/abs/1809.11096">BigGAN</a>). Currently focusing on getting a small prototype up and running and then potentially work with a lab/company to build this out!

Interested to learn about this project? <a href="mailto:sri.anumakonda06@gmail.com">Send me an email here.</a>


Sample image:

![alt text](sample.gif)
