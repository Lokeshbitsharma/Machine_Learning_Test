# Plot utility

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score



class PlotUtility:
    
    def __init__(self,mlflow_instance= None):
        '''
        
        Parameters
        ----------
        mlflow_instance : mlflow object, optional
            In case parameters need to be logged, mlflow instance object has to be passed. The default is None.

        Returns
        -------
        None.

        '''
        self.mlflow = mlflow_instance
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    
    def plot_loss(self,history, label, n):
        '''
        Plot Loss Vs Epoch.

        Parameters
        ----------
        history : TYPE
            DESCRIPTION.
        label : TYPE
            DESCRIPTION.
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        plt.rcParams['figure.figsize'] = (12, 10)
        # Use a log scale on y-axis to show the wide range of values.
        plt.semilogy(history.epoch, history.history['loss'],
                     color=self.colors[n], label='Train ' + label)
        plt.semilogy(history.epoch, history.history['val_loss'],
                     color=self.colors[n], label='Val ' + label,
                     linestyle="--")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')      
        plt.legend()
        if self.mlflow is not None:
            plt.savefig("Loss Curve.png")
            # plt.show()
            self.mlflow.log_artifact("Loss Curve.png")
            plt.close()
      
      
    def plot_metrics(self, history):
        '''
        Plot Metric Curve

        Parameters
        ----------
        history : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        plt.rcParams['figure.figsize'] = (12, 10)
        
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
          name = metric.replace("_"," ").capitalize()
          plt.subplot(2,2,n+1)
          plt.plot(history.epoch, history.history[metric], color=self.colors[0], label='Train')
          plt.plot(history.epoch, history.history['val_'+metric],
                   color=self.colors[0], linestyle="--", label='Val')
          plt.xlabel('Epoch')
          plt.ylabel(name)
          if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
          elif metric == 'auc':
            plt.ylim([0.8,1])
          else:
            plt.ylim([0,1])      
          plt.legend()
          if self.mlflow is not None:
              plt.savefig("Metric Curve.png")
              # plt.show()
              self.mlflow.log_artifact("Metric Curve.png")
              plt.close()
        
        
    def plot_cm(self, labels, predictions, p=0.5):
        '''
        Plot COnfusion Matrix.

        Parameters
        ----------
        labels : TYPE
            DESCRIPTION.
        predictions : TYPE
            DESCRIPTION.
        p : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        None.

        '''
        plt.rcParams['figure.figsize'] = (12, 10)
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.legend()
        if self.mlflow is not None:
            plt.savefig("Confusion Matrix.png")
            # plt.show()
            self.mlflow.log_artifact("Confusion Matrix.png")
            plt.close()
      
    
     
    def print_classification_report(self, y_test, y_pred):
        '''
        Generate Classification Report.

        Parameters
        ----------
        y_test : TYPE
            DESCRIPTION.
        y_pred : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        print("CLASSIFICATION REPOST : \n",classification_report(y_test, y_pred))
        print ("CONFUSION MATRIX \n",confusion_matrix(y_test, y_pred))
        print("Accuracy \n", accuracy_score(y_test, y_pred))
    
    
    def plot_roc(name, labels, predictions, **kwargs):
        '''
        Plot ROC curve

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        labels : TYPE
            DESCRIPTION.
        predictions : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.


        '''
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
        plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.xlim([-0.5,20])
        plt.ylim([80,100.5])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.legend(loc='lower right')
        if self.mlflow is not None:
            plt.savefig("ROC Curve.png")
            # plt.show()
            self.mlflow.log_artifact("ROC Curve.png")
            plt.close()
  
  