import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-receve',
  templateUrl: './receve.component.html',
  styleUrls: ['./receve.component.css']
})
export class ReceveComponent {
  constructor(private router:Router){
    if(sessionStorage.getItem('isloggin')!='true'){
     sessionStorage.setItem('url','recevoir')
     this.router.navigate(['login'])
    }
  }

  submit (form: NgForm) {
console.log('suis la')
  }

}
