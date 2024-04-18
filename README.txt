=============================================================================================
Anisotropic physics-regularized interpretable machine learning of microstructure evolution

IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Melville, J., Yadav, V., Yang, L., Krause, A. R., Tonks, M. R., & Harley, J. B. (2024). Anisotropic physics-regularized interpretable machine learning of microstructure evolution. Computational Materials Science, 238, 112941.
=============================================================================================

DESCRIPTION:
	Anisotropic Physics-Regularized Interpretable Machine Learning Microstructure Evolution (APRIMME) is a general-purpose machine learning solution for grain growth simulations. In prior work, PRIMME employed a deep neural network to predict site-specific migration as a function of its neighboring sites to model normal, isotropic, grain growth behavior. This work aims to extend this method by incorporating grain boundary misorientation-based grain growth behavior. APRIMME is trained on anisotropic simulations created using the Monte Carlo-Potts (MCP) model. The results of this work are compared statistically using grain radius, number of sides per grain, mean neighborhood misorientations, and the standard deviation of triple junction dihedral angles, and are found to match in most cases. The exceptions are small and seem to be related to two causes: (1) the deterministic model of APRIMME is learning from the stochastic simulations of MCP, which seems to accentuate triple junction behaviors; and, (2) a bias against very small grains is made evident in a quicker decrease in grains than expected at the beginning of an APRIMME simulation. APRIMME is also evaluated for its general ability to capture anisotropic grain growth behavior by first investigating different test case initial conditions, including a circle grain, three grain, and hexagonal grain microstructures.
		
CONTRIBUTORS: 
	Joseph Melville [1], Vishal Yadav [2], Lin Yang [2], Amanda R Krause [3], Michael R Tonks [2], Joel B Harley [1]
	
AFFILIATIONS:
	1. University of Florida, SmartDATA Lab, Department of Electrical and Computer Engineering
	2. University of Florida, Tonks Research Group, Department of Material Science and Engineering
	3. Carnegie Mellon University, Department of Materials Science and Engineering

FUNDING SPONSORS:
	U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award \#DE-SC0020384
	U.S. Department of Defence through a Science, Mathematics, and Research for Transformation (SMART) scholarship


