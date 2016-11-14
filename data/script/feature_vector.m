% for k = 1 : 250
%     str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class1/%d.jpg',k);
%     [featureouput]=featureextraction_task2(str_e);
%     train_features(k,1:512) = featureouput;
% end;
% 
% for a = 1 : 250
%     str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class2/%d.jpg',k);
%     [featureouput]=featureextraction_task2(str_e);
%     train_features(k+a,1:512) = featureouput;
% end;
% 
% for b = 1 : 250
%     str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class3/%d.jpg',k);
%     [featureouput]=featureextraction_task2(str_e);
%     train_features(k+a+b,1:512) = featureouput;
% end;
% 
% for c = 1 : 250
%     str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class4/%d.jpg',k);
%     [featureouput]=featureextraction_task2(str_e);
%     train_features(k+a+b+c,1:512) = featureouput;
% end;
% 
% for d = 1 : 250
%     str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class5/%d.jpg',k);
%     [featureouput]=featureextraction_task2(str_e);
%     train_features(k+a+b+c+d,1:512) = featureouput;
% end;

for k = 251 : 300
    str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class1/%d.jpg',k);
    [featureouput]=featureextraction_task2(str_e);
    test_features(k-250,1:512) = featureouput;
end;

for a = 251 : 300
    str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class2/%d.jpg',k);
    [featureouput]=featureextraction_task2(str_e);
    test_features(k+a-500,1:512) = featureouput;
end;

for b = 251 : 300
    str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class3/%d.jpg',k);
    [featureouput]=featureextraction_task2(str_e);
    test_features(k+a+b-750,1:512) = featureouput;
end;

for c = 251 : 300
    str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class4/%d.jpg',k);
    [featureouput]=featureextraction_task2(str_e);
    test_features(k+a+b+c-1000,1:512) = featureouput;
end;

for d = 251 : 300
    str_e = sprintf('/Users/ykg2910/Documents/4th_year_projects/Assignment3/data/class5/%d.jpg',k);
    [featureouput]=featureextraction_task2(str_e);
    test_features(k+a+b+c+d-1250,1:512) = featureouput;
end;

%save('train_features.mat', 'train_features');
save('test_features.mat', 'test_features');