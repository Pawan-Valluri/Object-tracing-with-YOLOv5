import cv2
import math


class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class Tracker():
    def __init__(self, coords_lst = [[[10, 70],[400, 75], 'up']]):
        self.smaller_font_scale = 0.3
        self.smaller_font_thickness = 2
        self.smaller_font_size = list(cv2.getTextSize("test",cv2.FONT_HERSHEY_SIMPLEX, self.smaller_font_scale, self.smaller_font_thickness))
        self.smaller_font_size[0] = list(self.smaller_font_size[0])
        self.smaller_font_size[0][1] += 8
        self.coords_lst = coords_lst
        coords_lst = self.process_coords()
        self.paths_trace=[]
        self.paths_now = 0
        self.colors = Colors()
        self.path_count = 0


    def get_center(self, xyxy):
        return [int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2) ]


    def distxy(self, pt1, pt2):
        return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)


    def process_coords(self):
        coords_lst = self.coords_lst
        for coord in coords_lst:
            x1, y1, x2, y2 = coord[0][0], coord[0][1], coord[1][0], coord[1][1]
            m = (y2-y1)/(x2-x1)
            c = y2 - m*x2
            coord.append(m)
            coord.append(c)
            coord.append({})
        return coords_lst
    


    def check_area(self, xy, coords):
        if min(coords[0][0], coords[1][0]) < xy[0] and max(coords[0][0], coords[1][0]) > xy[0]:
            if coords[2] == "down":
                return True if xy[1] > (coords[3]*xy[0] + coords[4]) else False
            elif coords[2] == "up":
                return True if xy[1] < (coords[3]*xy[0] + coords[4]) else False
        return False


    def check_exit(self):
        for coords in self.coords_lst:
            mode = coords[2]
            for path_trail in self.paths_trace:
                if len(path_trail[1]) > 2:
                    if self.check_area(path_trail[1][-1], coords) and not self.check_area(path_trail[1][-2], coords):
                        if path_trail[2] in coords[5].keys() == 0:
                            coords[5][path_trail[2]] = 1
                        path_trail[0] = 10
        return 


    def track(self, det_current): # Each entry in det_current = [center(int(x), int(y)), str(c) or class name, conf]
        self.paths_now = len(det_current)
        paths_to_be_popped = []
        if len(self.paths_trace)==0: # This happens for the first frame
            for k in det_current:
                self.paths_trace.append([0, [k[0]], k[1], k[2]]) # All points are added as new paths
        else:
            for k in range(len(self.paths_trace)):
                min_dist = 10000
                for j in range(len(det_current)):
                    dist = self.distxy(self.paths_trace[k][1][-1], det_current[j][0])
                    if dist < min_dist:
                        min_dist =  dist
                        closest = j
                if min_dist < 30:
                    self.paths_trace[k][1].append(det_current[closest][0])
                    if self.paths_trace[k][3] < det_current[closest][2]:
                        self.paths_trace[k][2], self.paths_trace[k][3] = det_current[closest][1], det_current[closest][2]
                    det_current.pop(closest)
                else:
                    paths_to_be_popped.append(self.paths_trace[k])
                    self.paths_trace[k][0] += 1
                    
                    
            
            for k in paths_to_be_popped:
                if k[0] > 3:
                    self.paths_trace.remove(k)
            
            
            for k in self.paths_trace:
                while len(k[1]) > 5:
                    k[1].pop(0)
                                
            if len(det_current):
                for j in det_current:
                    self.paths_trace.append([0, [j[0]], j[1], j[2]]) # new path
                    self.path_count += 1
                    for coords in self.coords_lst:
                        if j[1] in coords[5].keys():
                            coords[5][j[1]] += 1
                        else:
                            coords[5][j[1]] = 1
        
        

    def counter(self, im0):
        paths_per_area = []
        for coord in range(len(self.coords_lst)):
            paths_per_area.append({})
            im0 = cv2.line(im0, self.coords_lst[coord][0], self.coords_lst[coord][1], thickness=2, color=[255,255,255])            
            for key in self.coords_lst[coord][5].keys():
                paths_per_area[coord][str(key)] = self.coords_lst[coord][5][key]
        # paths_per_area = [i[5] for i in coords_lst]
        self.check_exit()
        for coord in self.coords_lst:
            go_down = 0
            for cord in coord[5].keys():
                im0 = cv2.putText(im0, f"{str([int(cord)])}: {coord[5][cord]}", (coord[1][0] +10, coord[1][1]+go_down), cv2.FONT_HERSHEY_SIMPLEX, self.smaller_font_scale, self.colors(int(cord), True), self.smaller_font_thickness)
                go_down += self.smaller_font_size[0][1]
        im0 = cv2.putText(im0, f"Counts now: {self.paths_now}, Counts Total: {self.path_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [161, 165, 66], 2)
        im0 = cv2.putText(im0, f"Class count: {paths_per_area}", (10,30+self.smaller_font_size[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [161, 165, 66], 2)
        #print(self.paths_now, paths_per_area)
        for path_trail in self.paths_trace:
            if len(path_trail[1]) > 2:
                for l in range(1, len(path_trail[1])):
                    im0 = cv2.line(im0, path_trail[1][l-1], path_trail[1][l], color=self.colors(int(path_trail[2]), True), thickness=2)
        return im0, paths_per_area