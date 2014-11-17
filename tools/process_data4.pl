#!/usr/bin/perl

open(MYFILE, 'iris.data');
open(OUT, ">iris.out");
open(IN, ">iris.in");

while (<MYFILE>) {
    chomp;
    @words = split(",", $_);

    if($#words > 3){
	
	for my $i (0 .. ($#words-1)){
	    print IN "@words[$i] ";
	}
	print IN "\n";
	
	if(@words[$#words] eq "Iris-setosa"){
	    print OUT "1\n";
	}
	
	if(@words[$#words] eq "Iris-versicolor"){
	    print OUT "2\n";
	}
	
	if(@words[$#words] eq "Iris-virginica"){
	    print OUT "3\n";
	}
    }

}

close(MYFILE);
close(OUT);
close(IN);
