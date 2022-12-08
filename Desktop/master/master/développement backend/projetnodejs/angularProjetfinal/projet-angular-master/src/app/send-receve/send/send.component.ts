import { Router } from '@angular/router';
import { OnInit } from '@angular/core';
import { Component } from '@angular/core';

@Component({
  selector: 'app-send',
  templateUrl: './send.component.html',
  styleUrls: ['./send.component.css']
})
export class SendComponent implements OnInit {
constructor(private router:Router){
 if(sessionStorage.getItem('isloggin')!='true'){
  sessionStorage.setItem('url','send')
  this.router.navigate(['login'])
 }
 
 
}
  ngOnInit(): void {

    console.log(sessionStorage.getItem('isloggin'))
    console.log('suis la')
  }
}
