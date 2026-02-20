from src.collection import DatasetCollection, Task, Interactions, Rule

if __name__ == '__main__':
    print('Starting')

    paderborn_collection = DatasetCollection('configs/paderborn.yaml')

    paderborn_task = Task(
        target = 'fault_element',
        domain_factors = ('fault_size', 'fault_arrangement', 'condition', 'fault_combination', 'fault_mode', 'fault_characteristic'),
        defaults= Rule(
            fixed = {'fault_size':0, 'fault_arrangement':0, 'condition':0, 'fault_combination':0, 'fault_mode':0, 'fault_characteristic':0},
            resolve={
                'sampling_rate':paderborn_collection.get_all_filter_values('sampling_rate')
            }
        ),
        classes= {
            paderborn_collection.get_filter_value_from_description('fault_element', 'normal'):Rule(
                fixed={
                    'fault_size':0,
                    'fault_arrangement':0,  
                    'fault_combination':0, 
                    'fault_mode':0, 
                    'fault_characteristic':0
                },
                resolve={'sampling_rate':paderborn_collection.get_all_filter_values('sampling_rate')}
            )
        }
    )

    paderborn_filters = paderborn_collection.create_valid_filter_combinations(
        depends=('fault_size', 'fault_arrangement', 'condition', 'fault_combination','fault_mode','fault_characteristic'), 
        task=paderborn_task,
        fault_size=[0,4,5], fault_mode=0, fault_characteristic=0,fault_combination=0)
    
    for filter  in paderborn_filters:
        print(filter)


    cwru_collection = DatasetCollection('configs/cwru.yaml')


    cwru_task = Task(
        target='fault_element',
        domain_factors=("fault_size", "bearing_position", "condition"),
        defaults=Rule(
            fixed={'fault_size': 0, 'bearing_position': 0, 'condition': 0},
            resolve={
                'sampling_rate': [1, 2],
                'fault_position':cwru_collection.get_filter_value_from_description('fault_position', 'centered')
            }
        ),
        classes= {
            cwru_collection.get_filter_value_from_description('fault_element', 'normal'):Rule(
                fixed={
                    'fault_size':0
                },
                resolve={
                    'fault_position': cwru_collection.get_filter_value_from_description('fault_position', 'normal'),
                    'sampling_rate': cwru_collection.get_filter_value_from_description('sampling_rate', 48000)
                }
            )
        },
        #interactions=Interactions.from_dict({
        #    'bearing_position':{1:{'fault_size':[0,1]}}
        #}),
        class_interactions={cwru_collection.get_filter_value_from_description('fault_element', 'inner ring'):Interactions.from_dict({
            'bearing_position':{1:{'sampling_rate':1}}
        }),
        cwru_collection.get_filter_value_from_description('fault_element', 'outer ring'):Interactions.from_dict({
            'bearing_position':{1:{'sampling_rate':1}}
        }),
        cwru_collection.get_filter_value_from_description('fault_element', 'ball'):Interactions.from_dict({
            'bearing_position':{1:{'sampling_rate':1}}
        })}
    )


    cwru_filters = cwru_collection.create_valid_filter_combinations(cwru_task, ('fault_size', 'bearing_position', 'condition'), 
        fault_size=[0,4])
    

    for filter  in cwru_filters:
        print(filter)
