class Augmentation:
    """
    Augmentation dataset for diversifying text
    
    """
    
    
    def __init__(self, ds, training, ):
        """
        in all funkction is better to use a model which specialize for this, or we can lose a meaning of sentence
        """
        self.ds = ds
        self.training = training
        
        
    
    def synonymous_rep(self):
        """
        Synonymous replacement
        
        """
        
        
    def random_insrt(self):
        """
        Random insertion 
        """
        
    def random_swap(self):
        """
        Random Swap 
        """
        
    def random_deletion(self):
        
        """
        Random deletion
        """
    def back_translation(self):
        """
        Back_translation - translation to different language and back
        """
    
    def augment_setting():
        """
        Gets from user info what type of augmentation he would like to do
        """