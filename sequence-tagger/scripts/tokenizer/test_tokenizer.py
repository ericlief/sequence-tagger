from tokenizer import tokenize
import unittest


class TestTokenizer(unittest.TestCase):
    def test_numbers(self):
        #print(tokenize('2.4.22  1,3,4  1.33,33  33-33-33'))
        self.assertEqual(tokenize('2.4.22\n1,3,4\n1.33,33\n33-33-33\n99.99.99.99'), ['2.4.22', '1,3,4', '1.33,33', '33-33-33', '99.99.99.99'])
        self.assertEqual(tokenize('+420 77.655.6549'), ['+420 77.655.6549'])
        self.assertEqual(tokenize('Quinta-feira, 12:12:23, 12/2/2012'), ['Quinta', '-', 'feira', ',', '12:12:23', ',', '12/2/2012']) 
        self.assertEqual(tokenize('E.U.A., U.E.'), ['E.U.A.', ',', 'U.E.'])
        self.assertEqual(tokenize('A/C pág. 2, sec. 2.3.1'), ['A/C', 'pág.', '2', ',', 'sec.',  '2.3.1'])
        self.assertEqual(tokenize('Dr. Erique A. Lief, Ph.D, dra. Rocha, il.mo sr. Moraes.'), ['Dr.', 'Erique', 'A.', 'Lief', ',', 'Ph.D', ',', 'dra.', 'Rocha', ',', 'il.mo', 'sr.', 'Moraes', '.'])
        
  

#s = r"E.U.A., Il.mo  a/c a/c/ c/ ele. pág. 2, sec. 2.3.1 Sup... 12:12:23 12/2/2012 \'\`  10,10,10 33-33-33 100.0.1,33 ''Dr. Erique A. Lief, Ph.D'', dra. Rocha, il.mo sr. Moraes. telefone +420 77.655.65.49 +420 77 655 65 49 e ele... a/c Ilma./ilma./il.ma/Ilma N.Sra. Garcia' . esse grande cabrão!--que é Dra. Morções; seu João às costas! 10,00.34, '4:15 p.m.' eric23_lief28@seznam.cz"



if __name__ == '__main__':
    unittest.main()

