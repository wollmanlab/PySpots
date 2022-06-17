#!/usr/bin/env python
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--command_filename", type=str, dest="command_filename", default="C:/Repos/fluidics/base_commands_config.xml", action='store', help="path for config file to be saved")
    parser.add_argument("-nv", "--n_valves", type=int, dest="n_valves", default=4, action='store', help="")
    parser.add_argument("-np", "--n_pumps", type=int, dest="n_pumps", default=1, action='store', help="")
    parser.add_argument("-fr", "--flow_rate", type=float, dest="flow_rate", default=1, action='store', help="")
    parser.add_argument("-v", "--volumes", type=list, dest="volumes", default=[0.25,0.5,1,1.5,2,5], action='store', help="")
    parser.add_argument("-lbv", "--line_bubble_volume", type=float, dest="line_bubble_volume", default=0.1, action='store', help="")
    parser.add_argument("-lv", "--line_volume", type=float, dest="line_volume", default=0.25, action='store', help="")
    parser.add_argument("-clv", "--common_line_volume", type=float, dest="common_line_volume", default=0.5, action='store', help="")
    parser.add_argument("-cv", "--chamber_volume", type=float, dest="chamber_volume", default=0.5, action='store', help="")
    parser.add_argument("-olv", "--out_line_volume", type=float, dest="out_line_volume", default=0.5, action='store', help="")
    parser.add_argument("-fv", "--flush_volume", type=float, dest="flush_volume", default=1, action='store', help="")
    parser.add_argument("-tv", "--tbs_volume", type=float, dest="tbs_volume", default=2, action='store', help="")
    parser.add_argument("-wv", "--wbuffer_volume", type=float, dest="wbuffer_volume", default=2, action='store', help="")
    parser.add_argument("-tcv", "--tcep_volume", type=float, dest="tcep_volume", default=2, action='store', help="")
    parser.add_argument("-hv", "--hybe_volume", type=float, dest="hybe_volume", default=2.5, action='store', help="")
    parser.add_argument("-iv", "--ibuffer_volume", type=float, dest="ibuffer_volume", default=2, action='store', help="")
    parser.add_argument("-rt", "--rinse_time", type=int, dest="rinse_time", default=60, action='store', help="")
    parser.add_argument("-ht", "--hybe_time", type=int, dest="hybe_time", default=600, action='store', help="")
    args = parser.parse_args()
    
if __name__ == '__main__':   
    print(args)
    command_filename  = args.command_filename
    n_valves = args.n_valves
    n_pumps = args.n_pumps
    flow_rate = float(args.flow_rate)/60 # mL/s """"""
    volumes = args.volumes
    line_bubble_volume = args.line_bubble_volume
    line_volume = args.line_volume
    common_line_volume = args.common_line_volume
    chamber_volume = args.chamber_volume
    out_line_volume = args.out_line_volume
    flush_volume = args.flush_volume
    tbs_volume = args.tbs_volume
    wbuffer_volume = args.wbuffer_volume
    tcep_volume = args.tcep_volume
    hybe_volume = args.hybe_volume
    ibuffer_volume = args.ibuffer_volume
    rinse_time = args.rinse_time
    hybe_time = args.hybe_time

    Valve_Commands = {
        'IBuffer1':{'valve':"1",'port':"1"},
        'IBuffer2':{'valve':"1",'port':"2"},
        'TBS1':{'valve':"1",'port':"3"},
        'TBS2':{'valve':"1",'port':"4"},
        'WBuffer1':{'valve':"1",'port':"5"},
        'WBuffer2':{'valve':"1",'port':"6"},
        'TCEP':{'valve':"1",'port':"7"},
        'Valve2':{'valve':"1",'port':"8"},
        'Valve3':{'valve':"1",'port':"9"},
        'Valve4':{'valve':"1",'port':"10"},
        'Hybe1':{'valve':"2",'port':"1"},
        'Hybe2':{'valve':"2",'port':"2"},
        'Hybe3':{'valve':"2",'port':"3"},
        'Hybe4':{'valve':"2",'port':"4"},
        'Hybe5':{'valve':"2",'port':"5"},
        'Hybe6':{'valve':"2",'port':"6"},
        'Hybe7':{'valve':"2",'port':"7"},
        'Hybe8':{'valve':"2",'port':"8"},
        'Hybe9':{'valve':"2",'port':"9"},
        'Hybe10':{'valve':"2",'port':"10"},
        'Hybe11':{'valve':"3",'port':"1"},
        'Hybe12':{'valve':"3",'port':"2"},
        'Hybe13':{'valve':"3",'port':"3"},
        'Hybe14':{'valve':"3",'port':"4"},
        'Hybe15':{'valve':"3",'port':"5"},
        'Hybe16':{'valve':"3",'port':"6"},
        'Hybe17':{'valve':"3",'port':"7"},
        'Hybe18':{'valve':"3",'port':"8"},
        'Hybe19':{'valve':"3",'port':"9"},
        'Hybe20':{'valve':"3",'port':"10"},
        'Hybe21':{'valve':"4",'port':"1"},
        'Hybe22':{'valve':"4",'port':"2"},
        'Hybe23':{'valve':"4",'port':"3"},
        'Hybe24':{'valve':"4",'port':"4"},
        'Hybe0':{'valve':"4",'port':"5"},
        'Hybe25':{'valve':"4",'port':"5"},
        'NuclearStain':{'valve':"4",'port':"5"},
        'Waste':{'valve':"4",'port':"6"},
        'Empty':{'valve':"4",'port':"7"},
        'Empty':{'valve':"4",'port':"8"},
        'Empty':{'valve':"4",'port':"9"},
        'Empty':{'valve':"4",'port':"10"},
    }

    Pump_Commands = {}
    for volume in volumes:
       Pump_Commands['Normal_Flow_'+str(volume)+'mL'] = str(int(volume/flow_rate))
       Pump_Commands['Reverse_Flow_'+str(volume)+'mL'] = str(int(volume/flow_rate))
    
    out = []
    out.append('<?xml version="1.0" encoding="ISO-8859-1"?>')
    out.append('<kilroy_configuration num_valves="'+str(n_valves)+'" num_pumps="'+str(n_pumps)+'">')
    """ Valve Commands """
    out.append('<valve_commands>')
    for valve,position in Valve_Commands.items():
        if 'Empty' in valve:
            continue
        out.append('<valve_cmd name="'+valve+'">')
        out.append('<valve_pos valve_ID="'+str(position['valve'])+'" port_ID="'+str(position['port'])+'"/>')
        if int(position['valve'])>1:
            valve = 'Valve'+str(position['valve'])
            position = Valve_Commands[valve]
            out.append('<valve_pos valve_ID="'+str(position['valve'])+'" port_ID="'+str(position['port'])+'"/>')
        out.append('</valve_cmd>')
    out.append('</valve_commands>')
    """ Pump Commands """
    out.append('<pump_commands>')
    out.append('<pump_cmd name="Normal Flow">')
    out.append('<pump_config speed="10.0" direction="Forward"/>')
    out.append('</pump_cmd>')
    out.append('<pump_cmd name="Reverse Flow">')
    out.append('<pump_config speed="10.0" direction="Reverse"/>')
    out.append('</pump_cmd>')
    out.append('<pump_cmd name="Stop Flow">')
    out.append('<pump_config speed="0.0"/>')
    out.append('</pump_cmd>')
    out.append('</pump_commands>')
    """ Protocols """
    out.append('<kilroy_protocols>')
    """ Flow """
    for volume in volumes:
        out.append('<protocol name="Normal_Flow_'+str(volume)+'mL">')
        out.append('<pump duration="'+str(int(volume/flow_rate))+'">Normal Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>')
        out.append('</protocol>')
        out.append('<protocol name="Reverse_Flow_'+str(volume)+'mL">')
        out.append('<pump duration="'+str(int(volume/flow_rate))+'">Reverse Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>')
        out.append('</protocol>')
    """ Position """
    for valve,position in Valve_Commands.items():
        if 'Empty' in valve:
            continue
        out.append('<protocol name="'+valve+'">')
        out.append('<valve duration="4">'+valve+'</valve>')
        out.append('</protocol>')
    """ Quick Flush """
    out.append('<protocol name="Quick Flush All Tubes">')
    for valve,position in Valve_Commands.items():
        if 'Empty' in valve:
            continue
        out.append('<valve duration="4">'+valve+'</valve>')
        out.append('<pump duration="'+str(int(line_bubble_volume/flow_rate))+'">Reverse Flow</pump>')
        out.append('<pump duration="'+str(int(line_volume/flow_rate))+'">Normal Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>')
    out.append('<valve duration="4">TBS1</valve>')
    out.append('<pump duration="'+str(int(2*common_line_volume/flow_rate))+'">Normal Flow</pump>')
    out.append('<pump duration="0">Stop Flow</pump>')
    out.append('</protocol>')

    """  Flush """
    out.append('<protocol name="Flush All Tubes">')
    for valve,position in Valve_Commands.items():
        if 'Empty' in valve:
            continue
        out.append('<valve duration="4">'+valve+'</valve>')
        out.append('<pump duration="'+str(int((flush_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>')
    out.append('<valve duration="4">TBS1</valve>')
    out.append('<pump duration="'+str(int(2*(common_line_volume+chamber_volume+out_line_volume)/flow_rate))+'">Normal Flow</pump>')
    out.append('<pump duration="0">Stop Flow</pump>')
    out.append('</protocol>')

    """ Reverse Flush """
    out.append('<protocol name="Reverse Flush All Tubes">')
    out.append('<valve duration="4">Waste</valve>')
    out.append('<pump duration="'+str(int(2*(common_line_volume+chamber_volume+out_line_volume)/flow_rate))+'">Reverse Flow</pump>')
    out.append('<pump duration="0">Stop Flow</pump>')
    for valve,position in Valve_Commands.items():
        if 'Empty' in valve:
            continue
        if 'TBS' in valve:
            continue
        out.append('<valve duration="4">'+valve+'</valve>')
        out.append('<pump duration="'+str(int((flush_volume/flow_rate)))+'">Reverse Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>')
    out.append('</protocol>')

    """ For each Hybe """
    hybes = [valve for valve in Valve_Commands.keys() if 'Hybe' in valve]
    n_hybes = len(hybes)
    ibuffers = [valve for valve in Valve_Commands.keys() if 'IBuffer' in valve]
    wbuffers = [valve for valve in Valve_Commands.keys() if 'WBuffer' in valve]
    tbses = [valve for valve in Valve_Commands.keys() if 'TBS' in valve]
    ibuffer_chunk_size = len(hybes)/len(ibuffers)
    wbuffer_chunk_size = len(hybes)/len(wbuffers)
    tbs_chunk_size = len(hybes)/len(tbses)
    ibuffer_dict = {}
    wbuffer_dict = {}
    tbs_dict = {}
    for i,hybe in enumerate(hybes):
        ibuffer_dict[hybe] = ibuffers[int(i/ibuffer_chunk_size)]
        wbuffer_dict[hybe] = wbuffers[int(i/wbuffer_chunk_size)]
        tbs_dict[hybe] = tbses[int(i/tbs_chunk_size)]
        """ Strip + Image """
        out.append('<protocol name="Strip+Image_'+str(hybe)+'">')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((tbs_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TCEP
        out.append('<valve duration="4">TCEP</valve>')
        out.append('<pump duration="'+str(int((tcep_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((line_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(hybe_time)+'">Stop Flow</pump>') # STRIP
        out.append('<pump duration="'+str(int(((tbs_volume-line_volume)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # IBuffer
        out.append('<valve duration="4">'+ibuffer_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int(((ibuffer_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        out.append('<pump duration="'+str(int(((ibuffer_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((line_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>') 
        out.append('<pump duration="'+str(int(((line_volume/2)/flow_rate)))+'">Reverse Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>')
        out.append('</protocol>')


        """ Strip + Hybe + Image """
        out.append('<protocol name="Strip+Hybe+Image_'+str(hybe)+'">')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((tbs_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TCEP
        out.append('<valve duration="4">TCEP</valve>')
        out.append('<pump duration="'+str(int((tcep_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((line_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(hybe_time)+'">Stop Flow</pump>') # STRIP
        out.append('<pump duration="'+str(int(((tbs_volume-line_volume)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # WBuffer
        out.append('<valve duration="4">'+wbuffer_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((wbuffer_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # Hybe
        out.append('<valve duration="4">'+hybe+'</valve>')
        out.append('<pump duration="'+str(int(((hybe_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        out.append('<pump duration="'+str(int(((hybe_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # Wbuffer
        out.append('<valve duration="4">'+wbuffer_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((line_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(hybe_time)+'">Stop Flow</pump>') # HYBE
        out.append('<pump duration="'+str(int(((wbuffer_volume-line_volume)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((tbs_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # IBuffer
        out.append('<valve duration="4">'+ibuffer_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int(((ibuffer_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        out.append('<pump duration="'+str(int(((ibuffer_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((line_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>') 
        out.append('<pump duration="'+str(int(((line_volume/2)/flow_rate)))+'">Reverse Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>')
        out.append('</protocol>')

        """ Hybe + Image """
        out.append('<protocol name="Hybe+Image_'+str(hybe)+'">')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((tbs_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # WBuffer
        out.append('<valve duration="4">'+wbuffer_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((wbuffer_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # Hybe
        out.append('<valve duration="4">'+hybe+'</valve>')
        out.append('<pump duration="'+str(int(((hybe_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        out.append('<pump duration="'+str(int(((hybe_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # Wbuffer
        out.append('<valve duration="4">'+wbuffer_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((line_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(hybe_time)+'">Stop Flow</pump>') # HYBE
        out.append('<pump duration="'+str(int(((wbuffer_volume-line_volume)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((tbs_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # IBuffer
        out.append('<valve duration="4">'+ibuffer_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int(((ibuffer_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        out.append('<pump duration="'+str(int(((ibuffer_volume/2)/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="'+str(rinse_time)+'">Stop Flow</pump>')
        # TBS
        out.append('<valve duration="4">'+tbs_dict[hybe]+'</valve>')
        out.append('<pump duration="'+str(int((line_volume/flow_rate)))+'">Normal Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>') 
        out.append('<pump duration="'+str(int(((line_volume/2)/flow_rate)))+'">Reverse Flow</pump>')
        out.append('<pump duration="0">Stop Flow</pump>')
        out.append('</protocol>')
    out.append('</kilroy_protocols>')
    out.append('</kilroy_configuration>')
    print('Writing Config File')
    with open(command_filename, 'w') as f:
        for line in out:
            f.write(line+'\n')