import numpy as np
import matplotlib.pyplot as plt

#Anaheim
results1 =  [[13.81450862516186, 14.873617099962859, 7.094983243215428, 16.170649838647492, 9.62976375104802, 3.247903102806381, 4.876219149093277, 8.458228592669972, 10.035027757540265, 6.525740117132736, 13.062835923559696, 11.027555268223724, 9.371697800479202, 10.065389604162196, 16.63365686033487, 10.616415486413977, 17.656684135675313, 11.778569947373807, 6.602225817555876, 6.711425154584907, 19.377715104380155, 12.237634642260288, 5.8297289953074145, 9.510858388644241, 9.327641273123952, 12.523194059071052, 8.226781028269844, 20.6534506748689, 5.700017700881343, 9.09753803587055], 
            [14.31880157226367, 15.358578457563487, 6.876548143790946, 16.414007781305312, 10.301594421237912, 3.389113861744525, 4.999726713634246, 8.526412852685334, 9.62561776921751, 6.312781967907381, 13.479126571866198, 11.485421248895042, 9.427074880712272, 10.000017452632937, 16.1624625147102, 10.371253563007922, 17.51735600636473, 12.240712858003112, 6.73478857176101, 6.778660996383505, 19.30167983731348, 12.317814017224977, 5.873964789939906, 9.752235182721094, 9.728435770967877, 12.807600033627395, 8.245893427992652, 20.947420941734364, 5.431287200287593, 9.041858813047485], 
            [14.530594872498293, 14.46543254833218, 6.747305259626887, 16.315336270285172, 9.410794359899054, 3.490980355067164, 4.783206587324994, 8.423288489974265, 9.902802027934701, 6.534912603849369, 13.327132075579204, 11.688913460443384, 9.590533703670715, 10.008266976971152, 15.828933236342749, 10.441226789574516, 18.06679834179518, 11.799527882305055, 6.590967802271265, 6.765513297765412, 19.704640440952733, 12.098232344709157, 5.59836926124917, 9.364996889755965, 9.584328493725147, 12.465782110703186, 8.214554479760434, 20.78056757987778, 5.628265423639916, 8.967746461816684]]

OD_pairs1 = [(122, 410), (165, 324), (151, 338), (90, 399), (195, 323), (104, 233), (94, 298), (15, 259), (126, 320), (186, 376), (125, 320), (168, 300), (127, 264), (124, 301), (87, 385), (38, 320), (75, 392), (11, 253), (110, 263), (15, 227), (81, 208), (49, 274), (98, 330), (100, 362), (166, 394), (131, 408), (172, 288), (67, 401), (100, 328), (149, 362)]

#simple
results2 = [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300],
            [9.61894791, 19.88848257, 30.07495306, 38.24036198, 48.17447517, 60.4316026, 64.0510384, 82.27493774, 89.91322706, 103.97728151, 110.53327916, 122.72256897, 129.88848957, 140.40604623, 152.1787726, 160.40693896, 172.79511453, 181.26313423, 189.45924294, 199.69696328, 210.62762795, 216.12540329, 242.3407216, 243.68579746, 250.72237362, 254.9010798, 262.33388981, 307.78085463, 296.04460882, 299.73908526],
            [8.91768255, 20.09039697, 29.8510147, 38.41192699, 47.50203975, 56.8890436, 63.66505222, 75.47546262, 83.1810521, 91.63539626, 104.00767216, 113.81437216, 121.46544112, 132.25207834, 136.18808357, 152.97627179, 156.09496631, 166.43112885, 177.93299479, 187.28212193, 198.05066468, 206.78398452, 217.23559637, 222.7195474, 231.81658381, 243.91751009, 252.29578531, 260.79009665, 271.3668167, 283.45664872]]

OD_pairs2 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30)]

#Sioux
results3 = [[26.51396473917235, 38.935623639798955, 29.05621403943341, 20.0848099783983, 28.426930164140572, 31.3715881456912, 39.96720205729659, 32.43562091434851, 25.9256248165418, 23.30147024302945, 34.71350829524113, 18.0485823618375, 6.32159677454557, 28.426930164140572, 25.4204285276163, 29.01871362932617, 41.3130259626113, 28.49304700716704, 42.34540756830404, 11.314243455178739, 34.71350829524113, 28.5112716847768, 44.67875940373581, 20.230932016552067, 34.92188701861576, 34.22709473947757, 38.04839478484738, 28.426930164140572, 34.82900814458106, 38.28251359401765, 42.23532759640743, 25.9256248165418, 23.30147024302945, 39.08837923191346, 40.858852502999085, 29.05621403943341, 26.41131741574977, 42.24170526509191, 2.06222568721317, 17.052500572506332, 42.60281239306072, 31.3715881456912, 39.076916595781015, 31.64186090766841, 36.456562827439306, 40.72783058331503, 32.43562091434851, 16.61302417814337, 14.72951905398906, 32.25281885954984, 38.28251359401765, 19.40490333410742, 38.82756405314002, 42.60281239306072, 35.6978002250233, 33.9572350139262, 15.836846173047139, 20.172376965111, 7.0429756993030095, 26.0969751396787, 32.69409686297805, 42.2464273336453, 32.221002708744315, 44.028338138808266, 36.02921413767392, 43.97589280750453, 35.45152931669462, 40.67006865352787, 34.56819296580762, 22.81051395556502, 40.465253577647246, 5.22806056298959, 3.02279654368237, 35.17631945139706, 29.102678751157022, 35.01376848960156, 27.426441337364402, 13.7223702825054, 31.9940267830031, 10.7294735255526, 41.25322380734045, 46.0232201613032, 27.50764586999069, 31.77950819128072, 26.0969751396787, 31.9940267830031, 28.5112716847768, 28.49304700716704, 34.71350829524113, 32.43562091434851, 44.67875940373581, 15.836846173047139, 24.406751008519933, 36.51218994688224, 28.97525117791354, 28.712674172245748, 38.04839478484738, 36.02921413767392, 26.51396473917235, 43.9869925447424], 
            [26.816554871403714, 38.7749047549147, 28.333308221076955, 19.778167551513707, 28.222959770631636, 32.283714308684424, 40.82314733161144, 32.37547003076781, 26.24557582627599, 23.236723406050345, 34.42803326477119, 19.210446432517468, 6.09312812749211, 27.963817284774827, 24.956844845652604, 29.413030326335317, 41.32380598832889, 29.431182116016718, 43.12342583449029, 11.360364745126121, 34.79598743011397, 28.505163932627987, 44.92484184442462, 20.893909137662913, 35.397049404055444, 34.65060221426869, 37.56363268817336, 28.563895701098964, 34.15693159064666, 38.58531822654828, 43.07757541302763, 25.616573728036357, 23.294005169715675, 39.86886689358441, 39.898500197851156, 28.71062364011699, 26.265276089089134, 41.72441144024603, 2.0520664288338746, 16.85812903196264, 42.55353042010068, 31.74415373358899, 38.67791859721028, 31.597974074858307, 36.39708230292502, 40.69811833671944, 32.633424094457624, 16.721293316452844, 14.094879952228366, 31.764301190667652, 39.072914050596154, 18.402637871396763, 38.394432998262744, 43.25450746306478, 36.70186570385659, 34.66886251691318, 15.092692433090136, 20.168317197885948, 7.049593373173824, 26.59128001385769, 33.13318249915017, 41.22535600789962, 32.523414375962965, 43.85315683008996, 36.130552372403514, 44.5977730675065, 34.958092696634765, 40.134994422438126, 34.977838229833786, 22.53738047406855, 40.15691538770293, 5.3629409436441655, 2.9595359979651805, 35.83661483516684, 29.486172601351395, 34.56750319318354, 26.693205988024296, 13.032877620565468, 31.352405788708623, 10.8427702927606, 40.9552681042524, 44.742384991183954, 26.635755042432677, 31.947064256114185, 25.61903303957992, 31.58806513015715, 28.197151902318215, 27.186264308224374, 34.50309793026163, 32.63922124029691, 44.44571311137066, 16.02431748380009, 24.694592884866168, 36.38569273010215, 28.847474611025014, 28.504327468225814, 37.277988177390675, 35.678344380629895, 26.365031961183874, 43.502269188300815], 
            [26.24882331111221, 39.11934497862416, 28.64879559999367, 20.35318520647054, 28.098656645468854, 31.385933555794782, 39.43573304035367, 32.78472851678804, 27.10372982236328, 23.077207902432814, 35.05180425147787, 17.928912493488834, 6.473262852136473, 28.5815879928571, 25.40852251766889, 29.33355153791071, 41.02372134776988, 28.507678668903786, 42.05117760311582, 11.252994350847164, 34.772742925267146, 28.942731239728023, 44.23703943840872, 19.911059489239975, 34.511968590743905, 33.57276467234722, 37.40670464192192, 28.6013556481079, 35.51681635895515, 38.01193371291378, 41.19886861656918, 25.02527708014972, 23.16065866056588, 39.63134495518407, 39.49657204308534, 29.4340968980957, 26.853696431863703, 42.90305680377563, 2.0664389391605997, 17.512933048945907, 42.70695990060729, 31.34622766640536, 39.51205963955331, 31.79815272998259, 36.61143025959407, 40.77923717821714, 32.85180144960682, 16.206986609447164, 14.891843887396739, 32.66166321611607, 38.22424234599071, 20.686014337578236, 38.13664748029482, 42.69741356365249, 35.57282188818986, 34.838579289192914, 15.469796758725597, 20.421392604889064, 7.022730053130953, 26.141121150486374, 33.1876096612259, 43.551866714392105, 32.54678600762343, 44.15999575031832, 35.859919060300776, 43.13731806631843, 35.42061663938032, 40.3678584830089, 34.13304387773771, 23.226912498936475, 40.626970891427334, 5.166235697152476, 2.9850173540397105, 35.693298870009166, 29.099249153587298, 34.88060144770753, 27.842877393390392, 14.19016965079452, 31.92599602681028, 10.65289976221163, 41.4944030050026, 46.35451552949245, 27.272377413674278, 30.43205055557199, 26.07634588760397, 31.868077688412836, 29.549981836030756, 28.1730143145923, 34.52537420160569, 32.37188303722595, 44.724537814742256, 15.674053712904751, 24.329549045137124, 37.57629947363659, 28.788428294559502, 28.883199515901186, 38.07321354739754, 35.721609132396885, 27.029591146654475, 42.85686005776107]]

OD_pairs3 = [(5, 19), (9, 23), (4, 14), (9, 15), (2, 22), (3, 14), (2, 18), (0, 22), (10, 14), (8, 12), (1, 23), (9, 18), (6, 19), (2, 22), (5, 15), (9, 12), (5, 23), (8, 21), (4, 20), (3, 12), (1, 23), (10, 16), (0, 21), (7, 16), (5, 16), (5, 21), (7, 13), (2, 22), (0, 17), (10, 20), (0, 16), (10, 14), (8, 12), (0, 19), (8, 22), (4, 14), (6, 23), (5, 22), (6, 17), (1, 12), (1, 18), (3, 14), (5, 14), (4, 16), (2, 20), (3, 20), (0, 22), (10, 12), (6, 16), (4, 17), (10, 20), (8, 14), (3, 19), (1, 18), (3, 18), (3, 16), (6, 18), (6, 14), (2, 12), (9, 13), (8, 20), (11, 16), (4, 22), (6, 12), (5, 18), (0, 18), (10, 17), (2, 21), (3, 17), (9, 21), (0, 20), (6, 15), (11, 12), (9, 22), (0, 13), (10, 21), (11, 13), (9, 14), (1, 15), (7, 15), (1, 20), (11, 15), (9, 19), (8, 13), (9, 13), (1, 15), (10, 16), (8, 21), (1, 23), (0, 22), (0, 21), (6, 18), (11, 22), (4, 19), (3, 23), (0, 23), (7, 13), (5, 18), (5, 19), (11, 18)]

#Sioux vs cov
results4 = [[44.028338138808266, 34.67962555195365, 5.22806056298959, 25.093988000949082, 20.0848099783983, 28.426930164140572, 32.5469818222843, 20.68380426641717, 27.01156381137603, 25.76734303000032, 42.003429673857255, 44.028338138808266, 42.34540756830404, 31.91273037831285, 31.292323406263698, 29.102678751157022, 38.4761777577053, 44.67875940373581, 42.003429673857255, 35.41865373532631, 39.96720205729659, 25.9256248165418, 44.67875940373581, 27.426441337364402, 20.82458616867579, 13.7223702825054, 34.92188701861576, 33.38242611876551, 42.003429673857255, 36.64988949790723], 
            [43.704512141732486, 34.66603152888729, 5.202391260423596, 25.167859956044264, 20.035379000714098, 28.580026454957643, 32.45510819177163, 20.720294425208095, 26.957995412562354, 25.75380305137854, 41.16586405753758, 43.044523325520686, 42.31821212558083, 31.528716979048763, 31.438969810659582, 29.12605906036467, 38.18667363647658, 44.25710500422905, 40.78900247703394, 34.923014398097216, 39.80082577405208, 25.951324612590593, 44.7568353127568, 27.60572276661713, 20.685779006003177, 13.706066437368682, 34.50293282287214, 33.32224841169254, 40.989760570896856, 36.510009657622255], 
            [42.94915515424681, 34.48482418392605, 5.229804670114328, 25.01123845013356, 20.05472019111685, 28.21290330796174, 31.694913584091946, 20.568635302278658, 26.89407807669223, 25.61903702397209, 41.61485274802889, 42.815434152100046, 42.349806425804914, 31.882686282179026, 31.45578340611059, 29.201021005820635, 38.36563190356264, 44.23818031347818, 41.43848836567292, 34.947914650572976, 39.44468317167829, 26.035777826996856, 43.95319643190398, 27.41596426517925, 20.445032768290407, 13.685294952307803, 34.63952796845901, 33.40936521841154, 41.947590214281156, 36.31200407583149]]

OD_pairs4 = [(6, 12), (5, 20), (6, 15), (2, 13), (9, 15), (2, 22), (6, 13), (11, 23), (9, 20), (8, 15), (2, 15), (6, 12), (4, 20), (7, 23), (4, 23), (0, 13), (7, 12), (0, 21), (2, 15), (4, 15), (2, 18), (10, 14), (0, 21), (11, 13), (3, 13), (9, 14), (5, 16), (4, 18), (2, 15), (11, 21)]

#Sioux vs d_real & cov
results5 = [[27.426441337364402, 22.81051395556502, 31.77950819128072, 20.172376965111, 25.67378992767408, 34.67962555195365, 27.01156381137603, 43.0969658854133, 23.731115413439518, 34.71350829524113], 
            [27.11193585417287, 22.62087026217914, 31.349189499441305, 19.887092466726905, 25.77693166853679, 34.62816684797879, 26.352808094722956, 43.16806252614206, 23.725334909803387, 34.72211256697711], 
            [27.927703348548437, 22.804498120656334, 31.312993598761523, 19.906630387758067, 25.602403978317167, 35.25533383603389, 26.43972144305246, 42.74973332268964, 23.822298501563015, 34.54025391438568]]
OD_pairs5 = [(11, 13), (9, 21), (8, 13), (6, 14), (7, 14), (5, 20), (9, 20), (2, 19), (8, 18), (1, 23)]

def GP4_plot(results, OD_pairs=None):
    results = np.array(results)
    sorted_indices = results[0].argsort()
    results = results[:,sorted_indices]
    # results = np.exp(results)
    # print(results)
    if OD_pairs is not None:
        OD_pairs = np.array(OD_pairs)
        OD_pairs = OD_pairs[sorted_indices]
        # print(OD_pairs)

    plt.figure()
    l1, = plt.plot(results[0], results[0], 'ko-', label='dijkstra')
    l2, = plt.plot(results[0], results[1], 'bo-', label='dijkstra_real')
    l3, = plt.plot(results[0], results[2], 'ro-', label='cov2')

    plt.legend()
    plt.show()

GP4_plot(results5, OD_pairs5)

# a = np.array(results5[2])-np.array(results5[1])
# b = np.where(a>0)[0]
# print(b)
# print(b.shape)
# print(np.array(results3[2][61])-np.array(results3[0][61]))
# print(OD_pairs3[61])