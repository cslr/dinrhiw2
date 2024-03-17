#!/usr/bin/perl

open(MYFILE, 'wdbc.data');
open(OUT, ">wdbc.out");
open(IN, ">wdbc.in");

while (<MYFILE>) {
    chomp;
    @words = split(",", $_);
    
    for my $i (2 .. $#words){
	print IN "@words[$i] ";
    }
    print IN "\n";
    
    if(@words[1] eq "B"){
	print OUT "0\n";
    }
    else{
	print OUT "1\n";
    }

}

close(MYFILE);
close(OUT);
close(IN);
