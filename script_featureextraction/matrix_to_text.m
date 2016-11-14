tic;
fid = fopen('data.txt', 'w+');
for i=1:size(output, 1)
    fprintf(fid, '%f ', output(i,:));
    fprintf(fid, '\n');
end
fclose(fid);
toc

tic;
csvwrite('data.txt', output);
toc;