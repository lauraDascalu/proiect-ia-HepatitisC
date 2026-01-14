import unittest
import random


from EvolutionaryAlgorithm import Chromosome, Crossover, Mutation

class TestEAModules(unittest.TestCase):
    def setUp(self):
        
        self.no_genes = 5
        self.min_vals = [0.0] * self.no_genes
        self.max_vals = [5.0] * self.no_genes
        
        #initializare cromozomi pentru test
        self.mother = Chromosome(self.no_genes, self.min_vals, self.max_vals)
        self.mother.genes = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        self.father = Chromosome(self.no_genes, self.min_vals, self.max_vals)
        self.father.genes = [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_crossover_arithmetic(self):
        
        rate = 1.0  # rata de crossover 100% 
        random.seed(42) # pentru reproductibilitate 
        
        child = Crossover.arithmetic(self.mother, self.father, rate)
        
        TOLERANCE = 1e-5
        for i in range(self.no_genes):
            m = self.mother.genes[i]
            f = self.father.genes[i]
            c = child.genes[i]

            a_i = (c - f) / (m - f)

         
            if abs(m - f) > TOLERANCE:
                a_i = (c - f) / (m - f)
                self.assertTrue(0.0 <= a_i <= 1.0, f"Ponderea a_i ({a_i}) la gena {i} nu este intre 0 si 1")
            else:
                self.assertAlmostEqual(c, m, delta=TOLERANCE)
                    
        print("\nTest reusit pentru clasa de incrucisare!")

    def test_mutation_swap(self):
       
        child = self.mother.__copy__()
        initial_genes = list(child.genes)
        rate = 1.0
        random.seed(10)

        Mutation.swap(child, rate)

        # identificam pozitiile unde genele s au schimbat
        diff_indices = [i for i in range(child.no_genes) if child.genes[i] != initial_genes[i]]

        # verificarea
        self.assertEqual(len(diff_indices), 2, "Swap ar trebui sa afecteze exact doua gene ")
        
        i1, i2 = diff_indices
        # verificam interschimbarea valorilor 
        self.assertEqual(child.genes[i1], initial_genes[i2], f"Valoarea de la indexul {i1} nu corespunde cu valoarea initiala de la {i2} ")
        self.assertEqual(child.genes[i2], initial_genes[i1], f"Valoarea de la indexul {i2} nu corespunde cu valoarea initiala de la {i1} ")
        
        # verificam restul genelor 
        for i in range(child.no_genes):
            if i not in diff_indices:
                self.assertEqual(child.genes[i], initial_genes[i], f"Gena {i} a fost modificata desi nu face parte din swap ")

        print("\nTest reusit pentru clasa de mutatie!")

if __name__ == '__main__':
    print("Rulare Teste Automate")
    unittest.main()