# AutoImplant 2020-2021


**Team members:**
+ [Mariya Donskova](https://github.com/maridonskova)
+ [Alexey Shevtsov](https://github.com/shevtsovalexey)
+ [Aleksandr Nevarko](https://github.com/AlexanderNevarko)
+ [Konstantin Soshin](https://github.com/SoshinK)
+ [Anita Soloveva](https://github.com/aniton)


## Installation

Execute from the directory you want the repo to be installed:

```
git clone https://github.com/aniton/AutoImplant-2020-2021.git
cd AutoImplant-2020-2021
pip install -e .
```

### Data Generation

+ Run to generate skulls with synthetic (cubic and spheric) defects of different size and localisation (one for each complete_skull):

```
python ./data_generation/synthetic_defect_generator.py
```
+ Run to generate triplets (comlete_skull, defective_skull, implant), registered to another triplets (currently 10 random skulls are selected for registration):

```
python ./data_generation/each_other_registration.py --n_triplets 10 --zone 'bilateral' 
```
This script will generate 225 additional triplets (each registration takes ~ 3 min). Available zones are: 'bilateral', 'frontoorbital', 'parietotemporal', 'random_1' and 'random_2'.
