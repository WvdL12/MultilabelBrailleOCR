# Braille recognition models and performance results

This directory contains performance results, model predictions and additional analysis artifacts.
Results are separated into a `J.UCS` subdirectory for the results reported in the journal article, and an `SU` directory for the final set of results with additional tests and models submitted in the Masters Thesis.
The results included in `J.UCS` are effectively a complete subset of those in the `SU` directory, but are kept separate to facilitate comparison with the discussions in the article, avoiding the potential confusion by including results not discussed in the article.
Lastly, an `Extra` directory is included containing results and preliminary analyses that were explored, but ultimately not utilised in either the article or the Masters thesis.

## Performance results

The results can be grouped together by common analyses.

- The set of files with the pattern `all_cbr_<test_set>_predictions.csv` include the model predictions for each sample in the respective test set. The `test` and `ood` test sets correspond to the ID [1] and OOD [2] test sets reported in the article and thesis, while the `abc_ood` [3], `b37_ood` [4] and `bcd_ood` [5] sets correspond to supplementary datasets that were ultimately not included in the final study and analyses.
- Different `metrics.csv` files include the performance metrics evaluated over the different evaluation sets, including ID, OOD and experimental sets (see next bullet). This includes overall performance, as well as per class or per label performance of different metrics.
- The results on various experimental augmentation test sets are included in the `<experiment_name>_experiment_<versioning>[_overall].csv` files. This includes the `brightness`, `noise`, `rotation` and `blur` experiments, with different versions exploring performance over slightly modified ranges of augmentation intensities. The `overall` performance denotes the performance with respect to all model predictions over all intensities, while the non-overall performance is evaluated with respect to model predictions for each intensity level individually.
- Lastly, some additional analyses done include additional datasets evaluated, as mentioned above, as well as experimental evaluation of model predictions, and subsequent performance, on either completely noisy or completely blank inputs. This was intended to explore the model prediction behaviour under inputs that do not correspond to any seen conditions. Blank inputs are associated with the empty Braille character, or class 0, and model predictions on such samples could allow insight into underlying biases, or input sensitivities, of models.

## References

[1] Ovodov, I.G. (2021a). Optical Braille recognition using object detection CNN. In: Pro￾ceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1741–1748. IEEE Computer Society, Los Alamitos, CA, USA.
[2] Li, R., Liu, H., Wang, X. and Qian, Y. (2018). DSBI: double-sided Braille image dataset and algorithm evaluation for Braille dots detection. In: Proceedings of the 2nd International Conference on Video and Image Processing, ICVIP ’18, pp. 65–69.  ssociation for Computing Machinery, New York, USA.
[3] Elaraby, N., Barakat, S. and Rezk, A. (2024). A generalized ensemble approach based on transfer learning for Braille character recognition. Information Processing and Management, vol. 61, no. 1.
[4] Gezahegn, H., Su, T.-Y., Su, W.-C. and Ito, M. (2019). An optical Braille recognition system for enhancing Braille literacy and communication between the blind and non-blind. Review of Undergraduate Computer Science, vol. 2018, no. 2, pp. 1–5.
[5] Kumaravelan, U. (2019). Braille character dataset. Published dataset. Available at: <https://www.kaggle.com/datasets/shanks0465/braille-character-dataset>
