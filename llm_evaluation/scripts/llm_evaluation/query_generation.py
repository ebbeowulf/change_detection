def room_likelihood(map_summary, tgt_class, include_furniture=False, include_relationships=False):
    roomType = map_summary['room'].replace("I am in a ","")[:-1]
    o_list = map_summary['furniture'].replace("There is ","")[:-1]

    stmt=f"What is the likelihood of a {tgt_class} being located in a {roomType}"
    if include_furniture:
        stmt+=" containing " + o_list 
    stmt+="?"
    if include_relationships:
        stmt+=map_summary['furniture_relationships']

    stmt+=" Return a single percentage as a JSON object."
    return stmt

def object_likelihood(map_summary, tgt_class, dkey, include_room=True):
    if len(map_summary['object_results'][dkey]['object_list'])==0:
        return None
    
    if include_room:
        stmt=map_summary['room']+map_summary['furniture']+map_summary['furniture_relationships']
    else:
        stmt=map_summary['furniture']+map_summary['furniture_relationships']
    stmt+=map_summary['object_results'][dkey]['combined_statement']
    
    stmt=stmt.replace("next to none of the objects","away from all of these objects")
    stmt+=f"Please quantify the likelihood of a {tgt_class} being in location {map_summary['object_results'][dkey]['object_list'][0]}"
    if len(map_summary['object_results'][dkey]['object_list'])>1:
        for val in map_summary['object_results'][dkey]['object_list'][1:-1]:
            stmt+=f", {val}"
        stmt+=f" or {map_summary['object_results'][dkey]['object_list'][-1]}"
    stmt+=". Return only a JSON object with a percentage between 0 and 1 assigned to each location id. Do not include any explanations, examples, or sample code."
    return stmt