import os
import json
import numpy as np

ROOT_DIR='/home/long/mmsegmentation/data/cityscapes'
fp = open(os.path.join(ROOT_DIR,'gtFine','cityscapes_panoptic_train.json'))
panopdict=json.load(fp)

instance_categories=np.array([])
instance_category_ids=np.array([])
for i in panopdict['categories']:
    if i['isthing']: 
        instance_categories=np.append(instance_categories,i['name'])
        instance_category_ids=np.append(instance_category_ids,i['id'])
plural_categories=np.array(['people', 'riders', 'cars', 'trucks', 'busses', 'trains', 'motorcycles', 'bicycles'])

    
def instances2prompt(instance_count):
    string=''
    total_classes=np.count_nonzero(instance_count)
    num_classes=0 #initializing
    for i,count in enumerate(instance_count):
        if count!=0:
            if count == 1:
                string+=str(1)+' '+instance_categories[i]
            elif count>1:
                string+=str(count)+' '+plural_categories[i]
            #add separators:
            num_classes+=1
            if num_classes<total_classes:
                string+=', ' if num_classes<total_classes-1 else ', and '  #add a comma for first classes, but 'and' before last class
     
    prompt = 'Street view with ' + string if string!='' else 'Streetview'
    # prompt += ', dark, dark skies, streetlights'
    return prompt

for set in ['val', 'train']:
    panoptic_path=os.path.join(ROOT_DIR,'gtFine','cityscapes_panoptic_'+set+'.json')
    fp = open(panoptic_path)
    panopdict=json.load(fp)
    with open('/home/long/ControlNet-for-BEP/training/cityscapes_instance_prompts1/'+set+'_prompts.json', 'w') as output_file:
        for index, image in enumerate(panopdict['annotations']):
            seg_info=image['segments_info']
            img_id=image['image_id']

            #count instance for each 'thing' class
            instance_count=np.zeros(len(instance_categories),dtype=np.uint8) #initialize to 0 instances
            for i in seg_info:
                instance_count[instance_category_ids==i['category_id']]+=1

            prompt=instances2prompt(instance_count)
            
            #for naming and constructing filepaths
            city=img_id.split('_')[0]
            source_filename=img_id+'_gtFine_labelTrainIds.png'
            target_filename=img_id+'_leftImg8bit.png'
            source=os.path.join(ROOT_DIR, 'gtFine', set, city, source_filename)
            target=os.path.join(ROOT_DIR, 'leftImg8bit', set, city, target_filename)
            line={"source": source, "target": target, "prompt": prompt}
            
            if index>0: output_file.write('\n')
            json.dump(line, output_file)