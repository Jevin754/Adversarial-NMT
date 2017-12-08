<span class="text-muted">Machine Translation Project:</span> Adversarial Neural Machine Translation
=============================================================

Neural Machine Translation (NMT) has become more and more popular in both academic and industry. Despite its success, the translation quality of NMT system is still unsatisfied and there remains a large room for improvement. The NMT model aims to maximize the probability of the target ground-truth sentence given the source sentence. Such an object does not guarantee the translation results to be natural and sufficient like human-translations. Some previous works tried to alleviate this limitation by reducing the objective inconsistency between NMT training and inference. Thus, they directly maximizing BLEU (Papineni et al., 2002). Some improvement is observed, but the objective still cannot bridge the gap between NMT translations and human-generated translations. Thanks to the success of Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), several works have been done to adopt GANs for NMT. The latest two works are (Lijun Wu et al., 2017) and (Zhen Yang et al., 2017).

In this project, we manage to investigate how Adversarial-NMT system works and implement an Adversarial-NMT model according to the work of (Lijun Wu et al., 2017). And finally compare our Adversarial-NMT with traditional NMT models. Due to limited computation resource and time, especially the GPU memories, we can only run our model on a small German to English dataset for some epochs, which consisting about 10K training sentence pairs and 3K validation pairs. 

To run the Adversarial-NMT, please use the following command:
```
python train_gan.py --data_file data/hw5 --optimizer Adam --batch_size 16 --model_file data --gpuid 0 --epochs 20
```