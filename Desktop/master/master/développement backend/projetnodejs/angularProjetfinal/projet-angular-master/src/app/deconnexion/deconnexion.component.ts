import { Router } from '@angular/router';
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-deconnexion',
  template: `
    <p>
      deconnexion works!
    </p>
  `,
  styleUrls: ['./deconnexion.component.css']
})
export class DeconnexionComponent implements OnInit {
  constructor (private router:Router){
    console.log('bonjour')
  }
  ngOnInit(): void {
sessionStorage.clear()
this.router.navigate([''])

  }


}
