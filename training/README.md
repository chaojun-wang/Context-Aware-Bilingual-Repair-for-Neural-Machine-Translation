USAGE INSTRUCTIONS
------------------

Scripts to training and evaluation of both DocRepair and Transference models.

1. ensure that all requirements are present, and that the path names in the ‘vars’ file are up-to-date. 

2. train a DocRepair model:

    ```
    bash scripts/train_docrepair.sh <directory_of_storing_model_checkpoints> <GPU_ID> <percentage_of_non-synthetic_training_data>
    ```

3. train a Transference model:

    ```
    bash scripts/train_transference.sh <directory_of_storing_model_checkpoints> <GPU_ID> <percentage_of_non-synthetic_training_data>
    ```

4. evaluate your model:

    ```
    bash scripts/evaluate.sh <directory_of_model_checkpoints> <prefix_of_evaluated_model_checkpoint> <beam_size> <conservative_penalty_weight> <conservative_penalty_way>
    ```

