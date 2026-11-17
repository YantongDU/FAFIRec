"""The catalog cache must preserve outputs and gradients of the equations."""
import unittest
import torch
from fafirec.model import FAFIRecModel
from fafirec.ml1m_model import PaperFAFIRecModel


class CatalogCacheTests(unittest.TestCase):
    def test_outputs_and_repeated_item_gradients_match_reference(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        features=torch.randn(15,31);features[0].zero_()
        for control in ['cf','uniform','diagnostic','diagnostic_no_alignment']:
            with self.subTest(control=control):
                reference=FAFIRecModel(14,features,d_model=8,max_len=5,n_heads=2,n_layers=1,dropout=0,control=control)
                cached=PaperFAFIRecModel(14,features,d_model=8,max_len=5,n_heads=2,n_layers=1,dropout=0,control=control)
                cached.load_state_dict(reference.state_dict())
                hist=torch.tensor([[0,0,1,2,3],[0,4,1,4,5]])
                pos=torch.tensor([[0,0,2,3,6],[0,1,4,5,6]])
                neg=torch.tensor([[0,0,9,9,10],[0,9,10,9,10]])
                diagnostic=torch.tensor([[.3,-.2],[-.5,.1]])
                reference.eval();cached.eval()
                a=reference(hist,pos,neg,diagnostic)
                b=cached(hist,pos,neg,diagnostic)
                for key in ['loss','rec_loss','align_loss']:
                    torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-6)
                a['loss'].backward();b['loss'].backward()
                for (name,p),(other,q) in zip(reference.named_parameters(),cached.named_parameters()):
                    self.assertEqual(name,other)
                    if p.grad is None:self.assertIsNone(q.grad)
                    else:torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-6)
                torch.testing.assert_close(reference.score_all(hist,diagnostic)[:,1:],cached.score_all(hist,diagnostic)[:,1:],rtol=1e-5,atol=1e-6)
                self.assertIsNone(cached._projected_catalog)
                self.assertEqual(reference.state_dict().keys(),cached.state_dict().keys())


if __name__=='__main__':unittest.main()
