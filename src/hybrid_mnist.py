import numpy as np

class Hybrid_MNIST(object):
    """
    params:
        both of these parameters apply if CNN and GP do NOT agree

        accept_cnn_tolerance: accept the CNN prediction if it is within this number of standard deviations from the GPs prediction
        double_criterion: 
            require that the predicted CNN class probability `p` is within `accept_cnn_tolerance` stddev's of both the same class in the GP _and_ the GP's top prediction
    """
    def __init__(self, accept_cnn_tolerance=0.5, stronger_criterion=False):
        self.accept_cnn_tolerance = accept_cnn_tolerance
        self.stronger_criterion = stronger_criterion
        self.name = "Hybrid model tol={0} stronger_crit={1}".format(accept_cnn_tolerance, self.stronger_criterion)

    def combine_predictions(self, cnn_probs, gp_mu, gp_var, verbose=False):
        assert (cnn_probs.shape[0] == gp_mu.shape[0] == gp_var.shape[0])
        decisions = []
        decision_probs = []
        decision_vars = []
        for (mu, var, cnn_probs) in zip(gp_mu, gp_var, cnn_probs):
            cnn_class = np.argmax(cnn_probs)
            gp_class = np.argmax(mu)
            
            gp_pred_prob = mu[gp_class]
            gp_pred_var = var[gp_class]
            
            cnn_pred_prob = cnn_probs[cnn_class]
            
            # both classes agree
            if gp_class == cnn_class:
                # we may have to accept the wrong decision but can't do anything about it
                decisions.append([0, gp_class, gp_pred_prob, gp_pred_var])
                decision_probs.append(mu)
                decision_vars.append(var)
            else:
                # disagreement! This is additional information
                # From prior experiments we suspect that NN is more likely to be correct [non-adverserial examples tested]
                # So, if we take the CNN prediction and check if it's the same as the _second_ highest GP prediction
                # try using that?

                # Revised:
                #  Take the CNN prediction IF it's probability is within 1 stddev of the corresponding GP class probability

                # core idea: if CNN is _too_ sure then we revert to GP prediction -- might be adverserial...?

                gp_prob_for_cnn_pred = mu[cnn_class]
                gp_stddev_for_cnn_pred = np.sqrt(var[cnn_class])

                if verbose:
                    print("Models disagree on predicted class")

                criterion = cnn_pred_prob < (gp_prob_for_cnn_pred + self.accept_cnn_tolerance*gp_stddev_for_cnn_pred)
                if self.stronger_criterion:
                    criterion = criterion and (cnn_pred_prob > (gp_pred_prob - self.accept_cnn_tolerance*np.sqrt(gp_pred_var)))


                if criterion:
                    if verbose:
                        print("  Taking CNN prediction, probability is within",
                              self.accept_cnn_tolerance, "stddev of GP probability")
                    decisions.append([1, cnn_class, cnn_pred_prob, -1])
                    decision_probs.append(cnn_probs)
                    decision_vars.append([-1 for _ in range(mu.shape[-1])])

                else:
                    if verbose:
                        print("  Taking GP prediction")
                    decisions.append([0, gp_class, gp_pred_prob, gp_pred_var])
                    decision_probs.append(mu)
                    decision_vars.append(var)
        return (np.array(decisions), np.array(decision_probs), np.array(decision_vars))
