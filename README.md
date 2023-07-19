# multitask_hyperbole_metaphor_detection
This repository contains the code for ACL2023 Findings paper: A Match Made in Heaven: A Multi-task Framework for Hyperbole and Metaphor Detection

**Abstract**: <br/>
Hyperbole and metaphor are common in day-to-day communication (e.g., "I am in deep trouble": how does trouble have depth?), which makes their detection important, especially in a conversational AI setting. Existing approaches to automatically detect metaphor and hyperbole have studied these language phenomena independently, but their relationship has hardly, if ever, been explored computationally. In this paper, we propose a multi-task deep learning framework to detect hyperbole and metaphor simultaneously. We hypothesize that metaphors help in hyperbole detection, and vice-versa. To test this hypothesis, we annotate two hyperbole datasets- HYPO and HYPO-L- with metaphor labels. Simultaneously, we annotate two metaphor datasets- TroFi and LCC- with hyperbole labels. Experiments using these datasets give an improvement of the state of the art of hyperbole detection by 12%. Additionally, our multi-task learning (MTL) approach shows an improvement of up to 17% over single-task learning (STL) for both hyperbole and metaphor detection, supporting our hypothesis. To the best of our knowledge, ours is the first demonstration of computational leveraging of linguistic intimacy between metaphor and hyperbole, leading to showing the superiority of MTL over STL for hyperbole and metaphor detection. Please find the arxiv preprint of the paper from [here] (https://arxiv.org/abs/2305.17480).

**Files**: <br/>
    MTL-F.ipynb  ----> Code for the multi-task learning with fully shared layer (MTL-F) approach <br/>
    STL.ipynb ----> Code for the single-task learning (STL) approach 

**Folder**: <br/>
    Data -----> Data curated and used for the experiments <br/>
	MTL-E ----> Code for the multi-task learning with encoder shared layer (MTL-E) approach

**How to run the code (MTL-F and STL)?** <br/>
    1. Open the required .ipynb file in Juypter notebook or Google Colab <br.>
    2. Add the required data from data.zip to data folder <br/>
    3. Follow the instructions in the notebook

**How to run the code for MTL-E?** <br/>
	1. Open the data_creation.ipynb file and run all the cells. <br/>
	2. Open the code_run.ipynb file and run all the cells.
	
**Note**: We have modified the code from the following repos for our work:
https://github.com/shahrukhx01/multitask-learning-transformers/tree/main/shared_encoder


